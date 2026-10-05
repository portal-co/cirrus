#![no_std]
extern crate alloc;

use alloc::vec::Vec;
use core::array;

pub use ml_kem::kem::common::rand_core::{CryptoRng, Rng, TryCryptoRng, TryRng};
use ml_kem::{
    Decapsulate, DecapsulationKey1024, Encapsulate, EncapsulationKey1024, Generate, SharedKey,
    kem::{Key, KeyExport},
    ml_kem_1024::Ciphertext,
};

const ML_KEM_Q: u16 = 3329;
const ML_KEM_1024_T_BYTES: usize = 4 * 384;
/// ML-KEM-1024 encapsulation-key encoding size from FIPS 203.
pub const ENCAPSULATION_KEY_BYTES: usize = 1568;
/// ML-KEM-1024 ciphertext encoding size from FIPS 203.
pub const CIPHERTEXT_BYTES: usize = 1568;
/// ML-KEM shared-key and payload-mask size.
pub const SHARED_KEY_BYTES: usize = 32;

/// Make a canonical, uniformly distributed (up to negligible reduction bias)
/// ML-KEM-1024 encapsulation key without generating or discarding a secret key.
///
/// The key is formed directly as `(t_hat || rho)`: 1024 independently sampled
/// coefficients in `[0, q)` are encoded with ML-KEM's 12-bit encoding and the
/// final 32 bytes are random `rho`. This deliberately does not retry parsing
/// random byte strings. The single parser call is an invariant check for our
/// locally constructed canonical encoding, not a rejection sampler.
fn public_key_without_secret(rng: &mut (dyn CryptoRng + '_)) -> EncapsulationKey1024 {
    let mut coefficients = [0u16; ML_KEM_1024_T_BYTES / 3 * 2];
    for coefficient in &mut coefficients {
        // Reducing a uniform 256-bit value modulo q has statistical distance
        // below q/2^256 per coefficient; this avoids variable-time/rejection
        // sampling while making the total key-distribution bias negligible.
        let mut wide = [0u8; 32];
        rng.fill_bytes(&mut wide);
        let residue = wide.iter().rev().fold(0u32, |acc, byte| {
            (acc * 256 + u32::from(*byte)) % u32::from(ML_KEM_Q)
        });
        *coefficient = residue as u16;
    }

    let encoded = Key::<EncapsulationKey1024>::from_fn(|index| {
        if index < ML_KEM_1024_T_BYTES {
            let pair = index / 3 * 2;
            let a = coefficients[pair];
            let b = coefficients[pair + 1];
            match index % 3 {
                0 => a as u8,
                1 => ((a >> 8) as u8) | ((b as u8) << 4),
                _ => (b >> 4) as u8,
            }
        } else {
            let mut byte = [0u8; 1];
            rng.fill_bytes(&mut byte);
            byte[0]
        }
    });

    EncapsulationKey1024::new(&encoded)
        .expect("handcrafted ML-KEM public key must have canonical coefficients")
}

/// Receiver setup for N-choice KEM OT. Exactly one generated decapsulation
/// key is retained for the selected index; all other public keys are
/// independently handcrafted and have no corresponding secret key.
pub fn miniot_start_recv<const N: usize>(
    rng: &mut (dyn CryptoRng + '_),
    choice: usize,
) -> ([EncapsulationKey1024; N], DecapsulationKey1024) {
    assert!(N > 0, "MiniOT must have at least one choice");
    let decaps = DecapsulationKey1024::generate_from_rng(rng);
    let selected = choice % N;
    let keys = array::from_fn(|index| {
        if index == selected {
            decaps.encapsulation_key().clone()
        } else {
            public_key_without_secret(rng)
        }
    });
    (keys, decaps)
}

/// Sender response: encapsulate independently to each public key and mask its
/// corresponding 32-byte payload with the KEM shared key.
pub fn miniot_sender<const N: usize>(
    rng: &mut (dyn CryptoRng + '_),
    payloads: [SharedKey; N],
    from_recv: [EncapsulationKey1024; N],
) -> [(Ciphertext, SharedKey); N] {
    array::from_fn(|index| {
        let (ciphertext, key) = from_recv[index].encapsulate_with_rng(rng);
        (
            ciphertext,
            SharedKey::from_fn(|byte| key[byte] ^ payloads[index][byte]),
        )
    })
}

/// Recover the payload at the selected index. The receiver retains the
/// selected decapsulation key for this operation; no private key is discarded.
pub fn miniot_finish_recv<const N: usize>(
    from_sender: [(Ciphertext, SharedKey); N],
    choice: usize,
    state: DecapsulationKey1024,
) -> SharedKey {
    assert!(N > 0, "MiniOT must have at least one choice");
    let (ciphertext, masked) = from_sender[choice % N];
    let key = state.decapsulate(&ciphertext);
    SharedKey::from_fn(|byte| key[byte] ^ masked[byte])
}

/// Errors decoding the fixed-size MiniOT handshake messages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MiniOtDecodeError {
    /// Message length did not match its fixed-size encoding.
    InvalidLength { expected: usize, found: usize },
    /// An untrusted encapsulation key failed ML-KEM's one-shot key check.
    InvalidEncapsulationKey { index: usize },
}

/// Encode a fixed batch of encapsulation keys for the MiniOT receiver message.
pub fn encode_public_keys<const N: usize>(keys: &[EncapsulationKey1024; N]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(N * ENCAPSULATION_KEY_BYTES);
    for key in keys {
        encoded.extend_from_slice(key.to_bytes().as_slice());
    }
    encoded
}

/// Decode and validate exactly `N` encapsulation keys. Invalid bytes fail once;
/// parsing never retries or samples replacement keys.
pub fn decode_public_keys<const N: usize>(
    bytes: &[u8],
) -> Result<[EncapsulationKey1024; N], MiniOtDecodeError> {
    let expected = N
        .checked_mul(ENCAPSULATION_KEY_BYTES)
        .expect("const MiniOT key count overflow");
    if bytes.len() != expected {
        return Err(MiniOtDecodeError::InvalidLength {
            expected,
            found: bytes.len(),
        });
    }
    let mut keys = Vec::with_capacity(N);
    for index in 0..N {
        let start = index * ENCAPSULATION_KEY_BYTES;
        let encoded = Key::<EncapsulationKey1024>::from_fn(|offset| bytes[start + offset]);
        let key = EncapsulationKey1024::new(&encoded)
            .map_err(|_| MiniOtDecodeError::InvalidEncapsulationKey { index })?;
        keys.push(key);
    }
    Ok(keys
        .try_into()
        .unwrap_or_else(|_| unreachable!("validated fixed key count")))
}

/// Encode sender encapsulations and their XOR-masked payloads.
pub fn encode_responses<const N: usize>(responses: &[(Ciphertext, SharedKey); N]) -> Vec<u8> {
    let item_len = CIPHERTEXT_BYTES + SHARED_KEY_BYTES;
    let mut encoded = Vec::with_capacity(N * item_len);
    for (ciphertext, masked_payload) in responses {
        encoded.extend_from_slice(ciphertext.as_slice());
        encoded.extend_from_slice(masked_payload.as_slice());
    }
    encoded
}

/// Decode exactly `N` sender encapsulations and masked payloads.
pub fn decode_responses<const N: usize>(
    bytes: &[u8],
) -> Result<[(Ciphertext, SharedKey); N], MiniOtDecodeError> {
    let item_len = CIPHERTEXT_BYTES + SHARED_KEY_BYTES;
    let expected = N
        .checked_mul(item_len)
        .expect("const MiniOT response count overflow");
    if bytes.len() != expected {
        return Err(MiniOtDecodeError::InvalidLength {
            expected,
            found: bytes.len(),
        });
    }
    let mut responses = Vec::with_capacity(N);
    for index in 0..N {
        let start = index * item_len;
        let ciphertext = Ciphertext::from_fn(|offset| bytes[start + offset]);
        let masked = SharedKey::from_fn(|offset| bytes[start + CIPHERTEXT_BYTES + offset]);
        responses.push((ciphertext, masked));
    }
    Ok(responses
        .try_into()
        .unwrap_or_else(|_| unreachable!("validated fixed response count")))
}

/// Sender-side convenience for the byte transport: validate public keys,
/// encapsulate to each, and return the fixed-size encoded response.
pub fn miniot_sender_encoded<const N: usize>(
    rng: &mut (dyn CryptoRng + '_),
    payloads: [[u8; SHARED_KEY_BYTES]; N],
    public_keys: &[u8],
) -> Result<Vec<u8>, MiniOtDecodeError> {
    let keys = decode_public_keys::<N>(public_keys)?;
    let payloads = payloads.map(|payload| SharedKey::from_fn(|index| payload[index]));
    Ok(encode_responses(&miniot_sender(rng, payloads, keys)))
}

/// Receiver-side convenience for the byte transport.
pub fn miniot_finish_recv_encoded<const N: usize>(
    response: &[u8],
    choice: usize,
    state: DecapsulationKey1024,
) -> Result<[u8; SHARED_KEY_BYTES], MiniOtDecodeError> {
    let responses = decode_responses::<N>(response)?;
    let key = miniot_finish_recv(responses, choice, state);
    Ok(array::from_fn(|index| key[index]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::convert::Infallible;
    use ml_kem::kem::common::rand_core::{TryCryptoRng, TryRng};

    #[derive(Clone)]
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

    fn payload(value: u8) -> SharedKey {
        SharedKey::from_fn(|index| value.wrapping_add(index as u8))
    }

    #[test]
    fn both_choice_bits_recover_only_the_selected_payload() {
        for choice in 0..2 {
            let mut recv_rng = TestRng(0x5245_4356_0000_0000 | choice as u64);
            let mut sender_rng = TestRng(0x5345_4E44_0000_0000 | choice as u64);
            let (public_keys, secret_key) = miniot_start_recv::<2>(&mut recv_rng, choice);
            let sent_payloads = [payload(0x21), payload(0xA7)];
            let response = miniot_sender(&mut sender_rng, sent_payloads.clone(), public_keys);
            let received = miniot_finish_recv(response, choice, secret_key);
            assert_eq!(received, sent_payloads[choice]);
        }
    }

    #[test]
    fn handcrafted_public_keys_parse_once_without_rejection_sampling() {
        let mut rng = TestRng(0x4D4C_4B45_4D);
        for choice in 0..2 {
            let (keys, secret_key) = miniot_start_recv::<2>(&mut rng, choice);
            assert_eq!(keys[choice], *secret_key.encapsulation_key());
            // Decoy keys are directly formed as canonical byte strings and
            // differ from the generated selected key; there is no decoy SK.
            assert_ne!(keys[1 - choice], keys[choice]);
            let encoded = encode_public_keys(&keys);
            assert_eq!(decode_public_keys::<2>(&encoded).unwrap(), keys);
        }
    }

    #[test]
    fn malformed_key_messages_fail_without_retries() {
        assert_eq!(
            decode_public_keys::<2>(&[0; ENCAPSULATION_KEY_BYTES]),
            Err(MiniOtDecodeError::InvalidLength {
                expected: 2 * ENCAPSULATION_KEY_BYTES,
                found: ENCAPSULATION_KEY_BYTES,
            })
        );
        let mut noncanonical = [0u8; ENCAPSULATION_KEY_BYTES];
        // The first 12-bit coefficient is q=3329, which is outside FIPS
        // 203's canonical [0,q-1] encoding range.
        noncanonical[0] = 0x01;
        noncanonical[1] = 0x0D;
        assert!(matches!(
            decode_public_keys::<1>(&noncanonical),
            Err(MiniOtDecodeError::InvalidEncapsulationKey { index: 0 })
        ));
    }

    #[test]
    fn response_encoding_round_trips_and_rejects_wrong_lengths() {
        let mut rng = TestRng(0x5245_5350_4F4E_5345);
        let (keys, state) = miniot_start_recv::<2>(&mut rng, 1);
        let responses = miniot_sender(&mut rng, [payload(7), payload(19)], keys);
        let encoded = encode_responses(&responses);
        let decoded = decode_responses::<2>(&encoded).unwrap();
        assert_eq!(decoded, responses);
        assert_eq!(miniot_finish_recv(responses, 1, state), payload(19));
        assert_eq!(
            decode_responses::<2>(&encoded[..encoded.len() - 1]),
            Err(MiniOtDecodeError::InvalidLength {
                expected: 2 * (CIPHERTEXT_BYTES + SHARED_KEY_BYTES),
                found: encoded.len() - 1,
            })
        );
    }
}
