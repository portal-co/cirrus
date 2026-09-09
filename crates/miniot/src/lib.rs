#![no_std]

#[cfg(feature = "std")]
extern crate std;

pub mod transport;

use core::array;

use digest::Digest;
use ml_kem::{
    Decapsulate, DecapsulationKey1024, Encapsulate, EncapsulationKey1024, Generate, SharedKey,
    kem::common::rand_core::CryptoRng, ml_kem_1024::Ciphertext,
};
pub fn miniot_start_recv<const N: usize>(
    rng: &mut (dyn CryptoRng + '_),
    v: usize,
) -> ([EncapsulationKey1024; N], DecapsulationKey1024) {
    let decaps = DecapsulationKey1024::generate_from_rng(rng);
    // Each decoy must be a *structurally valid* encapsulation key or
    // `EncapsulationKey1024::new` rejects it — the FIPS 203 §7.2 modulus check
    // round-trips the 12-bit encoding, and uniformly random bytes satisfy it
    // with probability ~(3329/4096)^256 ≈ 2^-76, so rejection-sampling random
    // byte strings effectively never terminates (this was the ~1400s hang).
    //
    // Generate a real keypair and keep only its encapsulation key: that is a
    // valid, undecapsulatable-by-the-receiver decoy, found in one shot.
    let mut x = array::from_fn(|_| {
        let decoy = DecapsulationKey1024::generate_from_rng(rng);
        decoy.encapsulation_key().clone()
    });
    x[v % N] = decaps.encapsulation_key().clone();
    (x, decaps)
}
pub fn miniot_sender<const N: usize>(
    rng: &mut (dyn CryptoRng + '_),
    v: [SharedKey; N],
    from_recv: [EncapsulationKey1024; N],
) -> [(Ciphertext, SharedKey); N] {
    return array::from_fn(|i| {
        let a = &from_recv[i];
        let v = &v[i];
        let (c, k) = a.encapsulate_with_rng(rng);
        (c, SharedKey::from_fn(|i| k[i] ^ v[i]))
    });
}
pub fn miniot_finish_recv<D: Digest + Clone, const N: usize>(
    from_sender: [(Ciphertext, SharedKey); N],
    v: usize,
    state: DecapsulationKey1024,
) -> SharedKey {
    let (a, b) = from_sender[v % N];
    let a = state.decapsulate(&a);
    let a = SharedKey::from_fn(|i| a[i] ^ b[i]);
    return a;
}
