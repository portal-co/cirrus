#![no_std]

use core::array;

use digest::{Digest, array::Array};
use ml_kem::{
    Decapsulate, DecapsulationKey1024, Encapsulate, EncapsulationKey1024, Generate, SharedKey,
    kem::common::rand_core::CryptoRng, ml_kem_1024::Ciphertext,
};
use rand::RngExt;
pub fn miniot_start_recv(
    rng: &mut (dyn CryptoRng + '_),
    v: bool,
) -> ([(EncapsulationKey1024); 2], DecapsulationKey1024) {
    let decaps = DecapsulationKey1024::generate_from_rng(rng);
    let fake = loop {
        if let Ok(x) = EncapsulationKey1024::new(&Array::from_fn(|_| rng.random())) {
            break x;
        }
    };
    (
        if v {
            [fake, decaps.encapsulation_key().clone()]
        } else {
            [decaps.encapsulation_key().clone(), fake]
        },
        decaps,
    )
}
pub fn miniot_sender(
    rng: &mut (dyn CryptoRng + '_),
    v: [SharedKey; 2],
    from_recv: [EncapsulationKey1024; 2],
) -> ([(Ciphertext, SharedKey); 2]) {
    return array::from_fn(|i| {
        let a = &from_recv[i];
        let v = &v[i];
        let (c, k) = a.encapsulate_with_rng(rng);
        (c, SharedKey::from_fn(|i| k[i] ^ v[i]))
    });
}
pub fn miniot_finish_recv<D: Digest + Clone>(
    from_sender: [(Ciphertext, SharedKey); 2],
    v: bool,
    state: DecapsulationKey1024,
) -> SharedKey {
    let (a, b) = from_sender[if v{1}else{0}];
    let a = state.decapsulate(&a);
    let a = SharedKey::from_fn(|i| a[i] ^ b[i]);
    return a;
}
