// @pinnedness: unpinned
// @stability: very-unstable
//! @ai: assisted
//!
//! OT-extension compatibility: adapt `miniot`'s ML-KEM 1-of-2 OT to
//! `volar-spec`'s [`BaseOt`] trait, so the post-quantum KEM OT can serve as the
//! base OT for the IKNP correlated-OT extension (and anything else built on
//! `BaseOt`). This mirrors volar-mpc's `{Chou-Orlandi, ML-KEM} × {…}` OT
//! matrix: here the ML-KEM base OT bootstraps many OTs instead of running one
//! OT per input bit.
//!
//! The IKNP extension is base-OT-agnostic through [`BaseOt`]; supplying
//! [`MlkemBaseOt`] lets the κ=128 base OTs run over ML-KEM-1024 (post-quantum)
//! rather than Chou-Orlandi (classical). All extension logic lives in
//! `volar-spec`; this module only bridges the base OT.
//!
//! # RNG seam
//!
//! `miniot` needs a `rand_core::CryptoRng` (ML-KEM key generation is
//! CSPRNG-gated); `volar-spec` OT functions take a `SpecRng`. [`SpecAsCrypto`]
//! adapts a `SpecRng` to `CryptoRng` so the same deterministic harness RNG
//! drives both. **The harness RNG is not a CSPRNG** — this is correct for the
//! in-process / test extension flow; production must back the base OT with a
//! real `CryptoRng`.

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;

use digest::Digest;
use ml_kem::{DecapsulationKey1024, EncapsulationKey1024};
use rand::rand_core::{Infallible, TryCryptoRng, TryRng};
use volar_spec::SpecRng;
use volar_spec::ot::base_ot::BaseOt;

// ============================================================================
// RNG bridge: SpecRng -> CryptoRng
// ============================================================================

/// Adapt a `volar-spec` [`SpecRng`] to a `rand_core` [`CryptoRng`], so ML-KEM
/// key generation can draw from the same deterministic harness RNG the rest of
/// the OT stack uses.
///
/// # Security
///
/// `SpecRng` makes no cryptographic-strength guarantee (the spec crate is
/// dependency-free and the harness RNG is a splitmix-style PRNG). Marking it
/// `CryptoRng` is a *compatibility shim* for the deterministic in-process /
/// test extension flow. A production base OT must use a real OS/CSPRNG
/// `CryptoRng` (e.g. `rand::rngs::SysRng`) directly with `miniot`, not this
/// adapter.
pub struct SpecAsCrypto<'a, R: SpecRng>(pub &'a mut R);

impl<R: SpecRng> TryRng for SpecAsCrypto<'_, R> {
    type Error = Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok(self.0.next_u32())
    }
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        let hi = self.0.next_u32() as u64;
        let lo = self.0.next_u32() as u64;
        Ok((hi << 32) | lo)
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Infallible> {
        for chunk in dest.chunks_mut(4) {
            let w = self.0.next_u32().to_le_bytes();
            chunk.copy_from_slice(&w[..chunk.len()]);
        }
        Ok(())
    }
}

impl<R: SpecRng> TryCryptoRng for SpecAsCrypto<'_, R> {}

// ============================================================================
// ML-KEM base OT as volar-spec BaseOt
// ============================================================================

/// ML-KEM-1024 1-of-2 OT as a volar-spec [`BaseOt`], for use as the IKNP base
/// OT. Generic over the masking `Digest`.
///
/// Message flow (three moves, matching [`BaseOt`]):
/// 1. [`BaseOt::sender_setup`] — the sender has no setup in the KEM-OT shape,
///    so this emits a unit setup message and keeps no state.
/// 2. [`BaseOt::recv_start`] — the receiver generates a real decapsulation key,
///    places its encapsulation key at index `c` and an undecapsulatable decoy
///    at `1-c`, and sends both encapsulation keys (the choice is hidden).
/// 3. [`BaseOt::sender_payload`] — the sender encapsulates against both keys,
///    masks each `L`-byte payload with the KDF-expanded shared key, and sends
///    the two `(ciphertext, masked)` pairs.
/// [`BaseOt::recv_finish`] decapsulates the chosen index and unmasks.
pub struct MlkemBaseOt<D> {
    _d: PhantomData<D>,
}

/// Receiver state: the real decapsulation key and its index.
pub struct MlkemRecvState {
    decaps: DecapsulationKey1024,
    choice: usize,
}

/// Receiver → sender message: the two encapsulation keys (choice hidden).
#[derive(Clone)]
pub struct MlkemRecvMsg {
    pub eks: [EncapsulationKey1024; 2],
}

/// Sender → receiver message: per index, the ciphertext then the masked payload.
pub struct MlkemPayloadMsg {
    pub cts: [ml_kem::ml_kem_1024::Ciphertext; 2],
    pub masked: [Vec<u8>; 2],
}

fn kdf_expand<D: Digest>(key: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut ctr: u32 = 0;
    while out.len() < n {
        let mut h = D::new();
        h.update(key);
        h.update(ctr.to_le_bytes());
        out.extend_from_slice(&h.finalize());
        ctr += 1;
    }
    out.truncate(n);
    out
}

impl<D: Digest, const L: usize> BaseOt<L> for MlkemBaseOt<D> {
    type SenderState = ();
    type ReceiverState = MlkemRecvState;
    type SetupMsg = ();
    type RecvMsg = MlkemRecvMsg;
    type PayloadMsg = MlkemPayloadMsg;

    fn sender_setup<R: SpecRng>(_rng: &mut R) -> (Self::SenderState, Self::SetupMsg) {
        ((), ())
    }

    fn recv_start<R: SpecRng>(
        rng: &mut R,
        _setup: &Self::SetupMsg,
        c: bool,
    ) -> (Self::ReceiverState, Self::RecvMsg) {
        let choice = c as usize;
        let (eks, decaps) = crate::miniot_start_recv::<2>(&mut SpecAsCrypto(rng), choice);
        (MlkemRecvState { decaps, choice }, MlkemRecvMsg { eks })
    }

    fn sender_payload<R: SpecRng>(
        rng: &mut R,
        _state: &Self::SenderState,
        recv_msg: &Self::RecvMsg,
        m0: &[u8; L],
        m1: &[u8; L],
    ) -> Self::PayloadMsg {
        use ml_kem::Encapsulate;
        let msgs: [&[u8]; 2] = [m0, m1];
        let mut cts: Vec<ml_kem::ml_kem_1024::Ciphertext> = Vec::with_capacity(2);
        let mut masked: Vec<Vec<u8>> = Vec::with_capacity(2);
        let mut crng = SpecAsCrypto(rng);
        for i in 0..2 {
            let (ct, shared) = recv_msg.eks[i].encapsulate_with_rng(&mut crng);
            let ks = kdf_expand::<D>(shared.as_slice(), L);
            masked.push((0..L).map(|j| msgs[i][j] ^ ks[j]).collect());
            cts.push(ct);
        }
        let masked: [Vec<u8>; 2] = [masked.remove(0), masked.remove(0)];
        let cts: [ml_kem::ml_kem_1024::Ciphertext; 2] = [cts.remove(0), cts.remove(0)];
        MlkemPayloadMsg { cts, masked }
    }

    fn recv_finish(state: &Self::ReceiverState, payload: &Self::PayloadMsg) -> [u8; L] {
        use ml_kem::Decapsulate;
        let c = state.choice;
        let shared = state.decaps.decapsulate(&payload.cts[c]);
        let ks = kdf_expand::<D>(shared.as_slice(), L);
        let mut out = [0u8; L];
        for j in 0..L {
            out[j] = payload.masked[c][j] ^ ks[j];
        }
        out
    }
}

// ============================================================================
// Ferret seed-COT derivation over the post-quantum base OT
// ============================================================================

use volar_spec::ot::ferret::{FerretReceiverSeed, FerretSenderSeed};
use volar_spec::ot::iknp::iknp_cot_extend_base;

/// Derive the `m` Ferret seed correlated-OTs from an IKNP extension whose base
/// OT is the ML-KEM post-quantum OT.
///
/// Ferret (`ferret_extend`) needs `m` seed COTs `(Delta; q[i], u[i], w[i])`
/// satisfying `w[i] = q[i] XOR (u[i].Delta)`. The standard bootstrap runs the
/// kappa base OTs plus IKNP to make exactly that correlation; running the base
/// OT over [`MlkemBaseOt`] makes the whole Ferret stack post-quantum at the
/// base-OT layer — the OT-extension-compatibility bridge for Ferret, matching
/// the IKNP one above.
///
/// Returns the `(sender_seed, receiver_seed)` pair `ferret_extend` consumes.
pub fn ferret_seed_cots_mlkem<D: Digest, R: SpecRng>(
    rng_s: &mut R,
    rng_r: &mut R,
    m: usize,
) -> (FerretSenderSeed, FerretReceiverSeed) {
    // Receiver choice bits `u[i]`.
    let mut u = alloc::vec![false; m];
    for b in u.iter_mut() {
        *b = (rng_r.next_u32() & 1) == 1;
    }
    // The C-OT correlation Delta.
    let mut delta = [0u8; 16];
    for chunk in delta.chunks_mut(4) {
        chunk.copy_from_slice(&rng_s.next_u32().to_le_bytes()[..chunk.len()]);
    }

    // IKNP with the ML-KEM base OT: sender gets r0[i], receiver gets v[i]
    // with v[i] = r0[i] XOR (u[i].Delta). Map r0 -> q, v -> w.
    let (q_rows, w_rows) =
        iknp_cot_extend_base::<MlkemBaseOt<D>, D, R, 16>(rng_s, rng_r, &u, &delta);
    debug_assert_eq!(q_rows.len(), m);
    debug_assert_eq!(w_rows.len(), m);

    (
        FerretSenderSeed {
            delta,
            q: q_rows.into_iter().collect(),
        },
        FerretReceiverSeed {
            u,
            w: w_rows.into_iter().collect(),
        },
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use sha2::Sha256;
    use volar_spec::ot::iknp::iknp_cot_extend_base;

    /// splitmix64 — the same deterministic harness RNG volar-mpc uses.
    struct Splitmix(u64);
    impl SpecRng for Splitmix {
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (z ^ (z >> 31)) as u32
        }
    }

    /// The ML-KEM base OT round-trips a 16-byte payload through the BaseOt
    /// state-machine interface, for both choices.
    #[test]
    fn mlkem_base_ot_roundtrip() {
        for c in [false, true] {
            let m0 = [0x11u8; 16];
            let m1 = [0x22u8; 16];
            let mut rs = Splitmix(0xAAAA);
            let mut rr = Splitmix(0xBBBB);

            let (s_state, setup) = <MlkemBaseOt<Sha256> as BaseOt<16>>::sender_setup(&mut rs);
            let (r_state, recv_msg) =
                <MlkemBaseOt<Sha256> as BaseOt<16>>::recv_start(&mut rr, &setup, c);
            let payload = <MlkemBaseOt<Sha256> as BaseOt<16>>::sender_payload(
                &mut rs, &s_state, &recv_msg, &m0, &m1,
            );
            let got = <MlkemBaseOt<Sha256> as BaseOt<16>>::recv_finish(&r_state, &payload);
            assert_eq!(got, if c { m1 } else { m0 }, "choice {}", c);
        }
    }

    /// OT-extension compatibility: drive volar-spec's IKNP correlated-OT
    /// extension with the ML-KEM base OT, and check the C-OT correlation
    /// (receiver value equals r0 when its bit is 0, r0 XOR delta when 1).
    #[test]
    fn iknp_extension_with_mlkem_base() {
        const M: usize = 32;
        let mut rs = Splitmix(0x1234);
        let mut rr = Splitmix(0x5678);

        let mut receiver_bits = [false; M];
        for (i, b) in receiver_bits.iter_mut().enumerate() {
            *b = (i * 7 + 3) % 5 < 2; // arbitrary mixed pattern
        }
        let delta_msg = [0xABu8; 16];

        let (sender_r0, receiver_v) = iknp_cot_extend_base::<MlkemBaseOt<Sha256>, Sha256, _, 16>(
            &mut rs,
            &mut rr,
            &receiver_bits,
            &delta_msg,
        );

        assert_eq!(sender_r0.len(), M);
        assert_eq!(receiver_v.len(), M);
        for j in 0..M {
            if receiver_bits[j] {
                let mut want = sender_r0[j];
                for b in 0..16 {
                    want[b] ^= delta_msg[b];
                }
                assert_eq!(receiver_v[j], want, "bit {} = 1 → r0 XOR delta", j);
            } else {
                assert_eq!(receiver_v[j], sender_r0[j], "bit {} = 0 → r0", j);
            }
        }
    }

    /// Ferret support: bootstrap the Ferret MPCOT extension from seed COTs
    /// derived over the post-quantum ML-KEM base OT, and check the C-OT
    /// relation holds end-to-end.
    #[test]
    fn ferret_with_mlkem_base() {
        use volar_spec::ot::ferret::{FERRET_REG_TOY, ferret_extend};

        let p = FERRET_REG_TOY;
        let m = p.seed_cot_count(false);
        let mut rs = Splitmix(0xFE_22_11);
        let mut rr = Splitmix(0x88_33_44);
        let mut rng_f = Splitmix(0xDE_AD_99);

        let (ss, rseed) = ferret_seed_cots_mlkem::<Sha256, _>(&mut rs, &mut rr, m);
        let delta = ss.delta;

        let out = ferret_extend(&mut rng_f, p, &ss, &rseed);

        // Sender/receiver output COTs satisfy w = q XOR (x*Delta).
        let n_out = out.sender_out.len();
        assert_eq!(n_out, p.output_cot_count(false));
        for j in 0..n_out {
            let mut want = out.sender_out[j];
            if out.recv_x[j] {
                for b in 0..16 {
                    want[b] ^= delta[b];
                }
            }
            assert_eq!(out.recv_z[j], want, "output row {}", j);
        }
        // The kept next-seed COTs satisfy the relation too.
        for j in 0..out.sender_seed.q.len() {
            let mut want = out.sender_seed.q[j];
            if out.receiver_seed.u[j] {
                for b in 0..16 {
                    want[b] ^= delta[b];
                }
            }
            assert_eq!(out.receiver_seed.w[j], want, "seed row {}", j);
        }
    }
}
