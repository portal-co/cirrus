#![no_std]

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
pub fn miniot_sender<D: Digest>(
    rng: &mut (dyn CryptoRng + '_),
    v: SharedKey,
    from_recv: [EncapsulationKey1024; 2],
) -> ([(Ciphertext, SharedKey); 2], Array<u8, D::OutputSize>) {
    let mut h = D::new();
    return (
        from_recv.map(|a| {
            let (c, k) = a.encapsulate_with_rng(rng);
            h.update(&c);
            (c, SharedKey::from_fn(|i| k[i] ^ v[i]))
        }),
        {
            h.update(&v);
            h.finalize()
        },
    );
}
pub fn miniot_finish_recv<D: Digest + Clone>(
    from_sender: [(Ciphertext, SharedKey); 2],
    hash_from_sender: Array<u8, D::OutputSize>,
    state: DecapsulationKey1024,
) -> SharedKey {
    let mut h = D::new();
    for (a, _) in &from_sender {
        h.update(a);
    }
    for (a, b) in from_sender {
        let a = state.decapsulate(&a);
        let a = SharedKey::from_fn(|i| a[i] ^ b[i]);
        let mut h = h.clone();
        h.update(&a);
        if h.finalize() == hash_from_sender {
            return a;
        }
    }
    unreachable!()
}
