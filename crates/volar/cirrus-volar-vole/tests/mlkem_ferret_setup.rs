use std::{
    convert::Infallible,
    sync::mpsc::{Receiver, SyncSender, sync_channel},
    thread,
    time::Duration,
};

use cirrus_volar_vole::{
    VoleScheduleCursor, VoleScheduleTicket, mlkem_ferret_prover_with_params,
    mlkem_ferret_verifier_with_params,
};
use miniot::{TryCryptoRng, TryRng};
use sha3::{Digest, Sha3_256};
use volar_ir_common::{
    EpochId, IterationId, IterationManifest, OrdinalRange, ScheduleHasher, ScheduleLimits,
    ScheduleManifest, WorkerLane,
};
use volar_spec::ot::{
    ferret::{FERRET_REG_TOY, FerretParams},
    two_party::StackIo,
};

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

struct Sha3ScheduleHasher;

impl ScheduleHasher for Sha3ScheduleHasher {
    fn digest(&self, bytes: &[u8]) -> [u8; 32] {
        Sha3_256::digest(bytes).into()
    }
}

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

fn toy_schedule(iteration_count: usize) -> ScheduleManifest {
    let iterations = (0..iteration_count)
        .map(|index| {
            let mut iteration =
                IterationManifest::new(IterationId(index as u64), index as u64, WorkerLane(0));
            iteration.correlation_range = OrdinalRange::new(index as u64, 1);
            iteration
        })
        .collect();
    ScheduleManifest::plan_from_iterations(
        [3; 32],
        [4; 32],
        0x564f_4c45,
        iteration_count as u32,
        1,
        iterations,
    )
    .expect("toy schedule must be canonical")
}

#[test]
fn direct_mlkem_bootstrap_runs_separate_roles_through_ferret_refill() {
    let setup_params = FerretParams {
        n: 512,
        k: 32,
        t: 7,
    };
    let main_params = FERRET_REG_TOY;
    let commit_count = setup_params.n - main_params.seed_cot_count(false) + 1;
    let schedule = toy_schedule(commit_count);
    let schedule_digest = schedule
        .digest_with(ScheduleLimits::default(), &Sha3ScheduleHasher)
        .expect("toy schedule digest must be canonical");
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

    let prover_schedule = schedule.clone();
    let prover = thread::spawn(move || {
        let mut rng = TestRng(0x5052_4F56_4552);
        let mut cursor = VoleScheduleCursor::new(&prover_schedule, schedule_digest)
            .expect("prover schedule must validate");
        let mut session =
            mlkem_ferret_prover_with_params(&mut rng, setup_params, main_params, &mut prover_io)
                .expect("prover direct ML-KEM/Ferret setup failed");
        let shares = (0..commit_count)
            .map(|index| {
                let ticket = VoleScheduleTicket::from_manifest(
                    &prover_schedule,
                    schedule_digest,
                    EpochId(0),
                    IterationId(index as u64),
                )
                .expect("prover ticket must resolve");
                cursor
                    .accept(ticket)
                    .expect("prover schedule must be in order");
                let bit = index % 2 == 0;
                (bit, session.commit_input(&mut rng, &mut prover_io, bit))
            })
            .collect::<Vec<_>>();
        cursor.finish().expect("prover schedule must be complete");
        (shares, session.remaining(), cursor.accepted())
    });

    let verifier_schedule = schedule;
    let verifier = thread::spawn(move || {
        let mut rng = TestRng(0x5645_5249_4649_4552);
        let mut cursor = VoleScheduleCursor::new(&verifier_schedule, schedule_digest)
            .expect("verifier schedule must validate");
        let mut session = mlkem_ferret_verifier_with_params(
            &mut rng,
            setup_params,
            main_params,
            &mut verifier_io,
        )
        .expect("verifier direct ML-KEM/Ferret setup failed");
        let shares = (0..commit_count)
            .enumerate()
            .map(|(index, _)| {
                let ticket = VoleScheduleTicket::from_manifest(
                    &verifier_schedule,
                    schedule_digest,
                    EpochId(0),
                    IterationId(index as u64),
                )
                .expect("verifier ticket must resolve");
                cursor
                    .accept(ticket)
                    .expect("verifier schedule must be in order");
                session.commit_input(&mut rng, &mut verifier_io)
            })
            .collect::<Vec<_>>();
        cursor.finish().expect("verifier schedule must be complete");
        (
            shares,
            session.delta().delta[0],
            session.remaining(),
            cursor.accepted(),
        )
    });

    let (prover_shares, prover_remaining, prover_scheduled) =
        prover.join().expect("prover thread panicked");
    let (verifier_shares, delta, verifier_remaining, verifier_scheduled) =
        verifier.join().expect("verifier thread panicked");

    assert_eq!(prover_shares.len(), commit_count);
    assert_eq!(verifier_shares.len(), commit_count);
    assert_eq!(prover_scheduled, commit_count);
    assert_eq!(verifier_scheduled, commit_count);
    assert_eq!(prover_remaining, verifier_remaining);
    assert_eq!(
        prover_remaining,
        setup_params.n - main_params.seed_cot_count(false) + main_params.output_cot_count(false)
            - commit_count,
        "consuming through the setup pool must trigger a main-profile refill"
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
