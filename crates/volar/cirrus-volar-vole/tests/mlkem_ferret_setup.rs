use std::{
    convert::Infallible,
    sync::mpsc::{Receiver, SyncSender, sync_channel},
    thread,
    time::Duration,
};

use cirrus_volar_vole::{mlkem_ferret_prover, mlkem_ferret_verifier};
use miniot::{TryCryptoRng, TryRng};
use volar_spec::ot::{ferret::FERRET_REG_TOY, two_party::StackIo};

/// Deterministic test-only generator. Never use for protocol security.
struct TestRng(u64);

impl TryRng for TestRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Ok((value ^ (value >> 31)) as u32)
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok((u64::from(self.try_next_u32()?) << 32) | u64::from(self.try_next_u32()?))
    }

    fn try_fill_bytes(&mut self, destination: &mut [u8]) -> Result<(), Self::Error> {
        for chunk in destination.chunks_mut(4) {
            let bytes = self.try_next_u32()?.to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
        Ok(())
    }
}

impl TryCryptoRng for TestRng {}

struct MessageIo {
    tx: SyncSender<(u8, Vec<u8>)>,
    rx: Receiver<(u8, Vec<u8>)>,
}

impl StackIo for MessageIo {
    fn send(&mut self, tag: u8, payload: &[u8]) {
        self.tx
            .send((tag, payload.to_vec()))
            .expect("peer disconnected");
    }

    fn recv(&mut self, expected_tag: u8) -> Vec<u8> {
        let (tag, payload) = self
            .rx
            .recv_timeout(Duration::from_secs(30))
            .expect("timed out waiting for peer protocol message");
        assert_eq!(tag, expected_tag, "unexpected protocol message tag");
        payload
    }
}

#[test]
fn mlkem_iknp_bootstrap_runs_separate_roles_through_repeated_ferret_refills() {
    let params = FERRET_REG_TOY;
    let commit_count = params.output_cot_count(false) + 1;
    let (prover_tx, verifier_rx) = sync_channel(1);
    let (verifier_tx, prover_rx) = sync_channel(1);
    let mut prover_io = MessageIo {
        tx: prover_tx,
        rx: prover_rx,
    };
    let mut verifier_io = MessageIo {
        tx: verifier_tx,
        rx: verifier_rx,
    };

    let prover = thread::spawn(move || {
        let mut rng = TestRng(0x5052_4F56_4552);
        let mut session = mlkem_ferret_prover(&mut rng, params, &mut prover_io)
            .expect("prover ML-KEM/IKNP setup failed");
        let shares = (0..commit_count)
            .map(|index| {
                let bit = index % 2 == 0;
                (bit, session.commit_input(&mut rng, &mut prover_io, bit))
            })
            .collect::<Vec<_>>();
        (shares, session.remaining())
    });

    let verifier = thread::spawn(move || {
        let mut rng = TestRng(0x5645_5249_4649_4552);
        let mut session = mlkem_ferret_verifier(&mut rng, params, &mut verifier_io)
            .expect("verifier ML-KEM/IKNP setup failed");
        let shares = (0..commit_count)
            .map(|_| session.commit_input(&mut rng, &mut verifier_io))
            .collect::<Vec<_>>();
        (shares, session.delta().delta[0], session.remaining())
    });

    let (prover_shares, prover_remaining) = prover.join().expect("prover thread panicked");
    let (verifier_shares, delta, verifier_remaining) =
        verifier.join().expect("verifier thread panicked");

    assert_eq!(prover_shares.len(), commit_count);
    assert_eq!(verifier_shares.len(), commit_count);
    assert_eq!(prover_remaining, verifier_remaining);
    assert_eq!(
        prover_remaining,
        2 * params.output_cot_count(false) - commit_count,
        "consuming across a Ferret output boundary must trigger another refill"
    );
    for ((bit, prover), verifier) in prover_shares.iter().zip(&verifier_shares) {
        let expected = prover.v[0] + prover.u[0][0] * delta;
        assert_eq!(
            verifier.q[0], expected,
            "chosen-bit VOLE relation for {bit}"
        );
        assert_eq!(
            prover.u[0][0],
            volar_spec::field::Galois128(u128::from(*bit))
        );
    }
}
