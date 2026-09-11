//! GRAM ORAM host (interpreter-side full ORAM access over garbled labels):
//! write then read a block through the label layer, decoding the result.

use cipher::consts::U16;
use cirrus_volar_garble::{GramOramHost};
use hybrid_array::Array;
use volar_oram::OramTree;
use volar_spec::garble::{Garble, GlobalSecret, gram_decode_label};

const Z: usize = 4;
const B: usize = 8;

fn det_secret() -> GlobalSecret<U16> {
    GlobalSecret::new(Array::<u8, U16>::from_fn(|i| {
        (i as u8).wrapping_mul(37) | 1
    }))
}

fn det_garble(seed: u8) -> Garble<U16> {
    Garble {
        base: Array::<u8, U16>::from_fn(|i| seed.wrapping_add(i as u8)),
    }
}

// Deterministic splitmix64 test RNG.
struct DetRng(u64);
impl DetRng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

// Encode a u64 as 64 LSB-first labels + their bases.
fn encode_u64(secret: &GlobalSecret<U16>, v: u64, seed: u8) -> (Vec<volar_spec::garble::Eval<U16>>, Vec<Garble<U16>>) {
    let mut labels = Vec::with_capacity(64);
    let mut bases = Vec::with_capacity(64);
    for i in 0..64 {
        let base = det_garble(seed.wrapping_add(i as u8));
        let bit = (v >> i) & 1 == 1;
        labels.push(secret.encode(&base, bit));
        bases.push(base);
    }
    (labels, bases)
}

#[test]
fn gram_oram_write_then_read_roundtrip() {
    let secret = det_secret();
    let levels = 4;
    let num_addrs = 8u64;
    let mut host = GramOramHost::<U16, Z, B>::new(det_secret(), levels, num_addrs);
    let mut tree = OramTree::<Z, B>::new(levels);
    let mut rng = DetRng(0xC0FFEE);

    // Result-data bases (the deterministic base supply the garbler used).
    let data_bases: Vec<Garble<U16>> = (0..(8 * B)).map(|i| det_garble(0x90 + i as u8)).collect();

    // WRITE addr 5, data = 0xA5 repeated.
    let data_byte = 0xA5u8;
    let write_bits: Vec<bool> = (0..(8 * B))
        .map(|i| (data_byte >> (i % 8)) & 1 == 1)
        .collect();
    let (al, ab) = encode_u64(&secret, 5, 0x10);
    host.access(
        &mut tree,
        &al,
        &ab,
        Some(&write_bits),
        &mut |i| data_bases[i].clone(),
        &mut || rng.next(),
    );

    // READ addr 5 back.
    let (al2, ab2) = encode_u64(&secret, 5, 0x50);
    let read = host.access(
        &mut tree,
        &al2,
        &ab2,
        None,
        &mut |i| data_bases[i].clone(),
        &mut || rng.next(),
    );

    // Decode the returned data labels and reconstruct the bytes.
    assert_eq!(read.data_labels.len(), 8 * B);
    let mut got = [0u8; B];
    for (i, label) in read.data_labels.iter().enumerate() {
        let bit = gram_decode_label(label, &data_bases[i]);
        if bit {
            got[i / 8] |= 1 << (i % 8);
        }
    }
    assert_eq!(got, [0xA5u8; B], "read must return the written block");
}

#[test]
fn gram_oram_multiple_addrs() {
    let secret = det_secret();
    let levels = 4;
    let mut host = GramOramHost::<U16, Z, B>::new(det_secret(), levels, 8);
    let mut tree = OramTree::<Z, B>::new(levels);
    let mut rng = DetRng(0x1234);
    let data_bases: Vec<Garble<U16>> = (0..(8 * B)).map(|i| det_garble(0xA0 + i as u8)).collect();

    let bits_of = |byte: u8| -> Vec<bool> { (0..(8 * B)).map(|i| (byte >> (i % 8)) & 1 == 1).collect() };
    let read_byte = |host: &mut GramOramHost<U16, Z, B>, tree: &mut OramTree<Z, B>, addr: u64, rng: &mut DetRng, seed: u8| -> [u8; B] {
        let (al, ab) = encode_u64(&secret, addr, seed);
        let read = host.access(tree, &al, &ab, None, &mut |i| data_bases[i].clone(), &mut || rng.next());
        let mut got = [0u8; B];
        for (i, label) in read.data_labels.iter().enumerate() {
            if gram_decode_label(label, &data_bases[i]) {
                got[i / 8] |= 1 << (i % 8);
            }
        }
        got
    };

    // Write addr 0 = 0x11, addr 3 = 0x77.
    let (a0l, a0b) = encode_u64(&secret, 0, 0x20);
    host.access(&mut tree, &a0l, &a0b, Some(&bits_of(0x11)), &mut |i| data_bases[i].clone(), &mut || rng.next());
    let (a3l, a3b) = encode_u64(&secret, 3, 0x30);
    host.access(&mut tree, &a3l, &a3b, Some(&bits_of(0x77)), &mut |i| data_bases[i].clone(), &mut || rng.next());

    assert_eq!(read_byte(&mut host, &mut tree, 0, &mut rng, 0x40), [0x11; B]);
    assert_eq!(read_byte(&mut host, &mut tree, 3, &mut rng, 0x60), [0x77; B]);
}

// The **encrypted tree** (ported S5/S6 scheme): the tree at rest holds only
// ciphertext under the host's AES key, while reads/writes still round-trip.
#[test]
fn gram_oram_encrypted_tree_roundtrip() {
    let secret = det_secret();
    let levels = 4;
    let num_addrs = 8u64;
    let key = [0x42u8; 16];
    let mut host = GramOramHost::<U16, Z, B>::new_encrypted(det_secret(), levels, num_addrs, key, 16);
    let mut tree = OramTree::<Z, B>::new(levels);
    // Pre-format: the fresh (dummy) entries become ciphertext.
    host.cryptor().unwrap().format_tree(&mut tree);
    let mut rng = DetRng(0xBEEF);
    let data_bases: Vec<Garble<U16>> = (0..(8 * B)).map(|i| det_garble(0x70 + i as u8)).collect();
    let bits_of = |byte: u8| -> Vec<bool> { (0..(8 * B)).map(|i| (byte >> (i % 8)) & 1 == 1).collect() };

    // Write addr 2 = 0x5A, addr 6 = 0xC3.
    for (addr, byte, seed) in [(2u64, 0x5Au8, 0x10u8), (6, 0xC3, 0x20)] {
        let (al, ab) = encode_u64(&secret, addr, seed);
        host.access(&mut tree, &al, &ab, Some(&bits_of(byte)), &mut |i| data_bases[i].clone(), &mut || rng.next());
    }
    // The stored tree bytes are ciphertext: a written block's data is not the
    // plaintext byte pattern.
    let plaintext_hits = tree
        .buckets
        .iter()
        .flat_map(|b| b.entries.iter())
        .filter(|e| e.data == [0x5Au8; B] || e.data == [0xC3u8; B])
        .count();
    assert_eq!(plaintext_hits, 0, "encrypted tree must not store plaintext blocks");

    // Reads still round-trip.
    for (addr, byte, seed) in [(2u64, 0x5Au8, 0x30u8), (6, 0xC3, 0x40)] {
        let (al, ab) = encode_u64(&secret, addr, seed);
        let read = host.access(&mut tree, &al, &ab, None, &mut |i| data_bases[i].clone(), &mut || rng.next());
        let mut got = [0u8; B];
        for (i, label) in read.data_labels.iter().enumerate() {
            if gram_decode_label(label, &data_bases[i]) {
                got[i / 8] |= 1 << (i % 8);
            }
        }
        assert_eq!(got, [byte; B], "encrypted read addr {addr}");
    }
}
