#![no_std]

use core::array;

use digest::{Digest, array::Array};
use ml_kem::{
    Decapsulate, DecapsulationKey1024, Encapsulate, EncapsulationKey1024, Generate, SharedKey,
    kem::common::rand_core::CryptoRng, ml_kem_1024::Ciphertext,
};
use rand::RngExt;
pub fn miniot_start_recv<const N: usize>(
    rng: &mut (dyn CryptoRng + '_),
    v: usize,
) -> ([(EncapsulationKey1024); N], DecapsulationKey1024) {
    let decaps = DecapsulationKey1024::generate_from_rng(rng);
    let mut x = array::from_fn(|_| {
        loop {
            if let Ok(x) = EncapsulationKey1024::new(&Array::from_fn(|_| rng.random())) {
                break x;
            }
        }
    });
    x[v % N] = decaps.encapsulation_key().clone();
    (x, decaps)
}
pub fn miniot_sender<const N: usize>(
    rng: &mut (dyn CryptoRng + '_),
    v: [SharedKey; N],
    from_recv: [EncapsulationKey1024; N],
) -> ([(Ciphertext, SharedKey); N]) {
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
