//! Role-separated ML-KEM-1024 MiniOT → Ferret-Reg setup/main COT stack.
//!
//! The prover owns the COT receiver choices/values; the verifier owns the
//! correlated-OT `Delta`. Direct MiniOT establishes the Ferret-Reg setup seed,
//! then one official setup-parameter Ferret iteration creates enough COTs to
//! seed the official main-iteration profile. The direct KEM-to-COT bootstrap
//! is experimental and semi-honest; it is not a paper-standard bootstrap or a
//! production-reviewed cryptographic construction.

use alloc::vec::Vec;
use core::fmt;

use cipher::consts::U1;
use cirrus_core::Pusher;
use hybrid_array::Array;
use miniot::{
    CIPHERTEXT_BYTES, CryptoRng, ENCAPSULATION_KEY_BYTES, MiniOtDecodeError, SHARED_KEY_BYTES,
};
use volar_spec::{
    SpecRng,
    field::Galois128,
    ot::{
        ferret::{
            CotPoolReceiver, CotPoolSender, FERRET_REG_MAIN, FERRET_REG_SETUP, FerretParams,
            FerretReceiverSeed, FerretSenderSeed, PoolSeedError, checked_seed_cot_count,
        },
        two_party::{
            StackIo, stack_bea95_receiver, stack_bea95_sender, stack_refill_receiver,
            stack_refill_sender,
        },
    },
    vole::{
        Delta, Q, Vope,
        setup::{vole_commit_bit_prover_share, vole_commit_bit_verifier_share},
    },
};

const TAG_MLKEM_COT_KEYS: u8 = 13;
const TAG_MLKEM_COT_RESPONSES: u8 = 14;
/// Keep each batched MiniOT message below the cloud helper's 128 KiB frame cap.
const MINIOT_BATCH_SIZE: usize = 16;
/// Bound the number of base COTs admitted by the parameterized test/profile API.
const MAX_SETUP_SEED_COTS: usize = 50_000;
/// Maximum one-time setup iteration size admitted by the Ferret-Reg profile.
const MAX_SETUP_PROFILE_COTS: usize = FERRET_REG_SETUP.n;
/// Maximum main iteration size admitted by the Ferret-Reg profile.
const MAX_MAIN_PROFILE_COTS: usize = FERRET_REG_MAIN.n;
/// Ferret-Reg's Table 2 semi-honest main seed COT count.
const MAX_MAIN_SEED_COTS: usize = 606_907;

/// Errors detected while establishing the direct ML-KEM/Ferret seed state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlKemFerretError {
    /// Ferret parameters or independently generated seed dimensions are invalid.
    FerretSeed(PoolSeedError),
    /// A fixed-size ML-KEM MiniOT message was malformed.
    MiniOt(MiniOtDecodeError),
    /// The setup seed exceeds the explicitly bounded MiniOT bootstrap limit.
    BootstrapTooLarge {
        /// Number of setup-seed COTs requested by the parameters.
        requested: usize,
        /// Maximum setup-seed COT count admitted by this API.
        maximum: usize,
    },
    /// A requested regular Ferret profile exceeds the researched Reg dimensions.
    ProfileTooLarge {
        /// Requested output count.
        requested: usize,
        /// Maximum output count admitted for this profile.
        maximum: usize,
    },
    /// The main profile's seed exceeds the researched Reg seed dimensions.
    MainSeedTooLarge {
        /// Requested main-profile seed count.
        requested: usize,
        /// Maximum main-profile seed count admitted by this API.
        maximum: usize,
    },
    /// The setup iteration cannot produce enough COTs for the main seed.
    SetupOutputTooSmall {
        /// Number of COTs required by the main profile's seed.
        required: usize,
        /// Total number of COTs produced by the setup profile.
        available: usize,
    },
    /// The verifier's Delta must be nonzero for the Quicksilver verifier relation.
    ZeroDelta,
}

impl fmt::Display for MlKemFerretError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FerretSeed(error) => write!(formatter, "invalid Ferret seed: {error:?}"),
            Self::MiniOt(error) => write!(formatter, "invalid ML-KEM MiniOT message: {error:?}"),
            Self::BootstrapTooLarge { requested, maximum } => write!(
                formatter,
                "ML-KEM bootstrap needs {requested} COTs; limit is {maximum}"
            ),
            Self::ProfileTooLarge { requested, maximum } => write!(
                formatter,
                "Ferret-Reg profile needs {requested} output COTs; limit is {maximum}"
            ),
            Self::MainSeedTooLarge { requested, maximum } => write!(
                formatter,
                "Ferret-Reg main seed needs {requested} COTs; limit is {maximum}"
            ),
            Self::SetupOutputTooSmall {
                required,
                available,
            } => write!(
                formatter,
                "Ferret setup produced {available} COTs; main seed needs {required}"
            ),
            Self::ZeroDelta => {
                formatter.write_str("verifier Delta was zero after bounded sampling")
            }
        }
    }
}

impl core::error::Error for MlKemFerretError {}

impl From<PoolSeedError> for MlKemFerretError {
    fn from(error: PoolSeedError) -> Self {
        Self::FerretSeed(error)
    }
}

impl From<MiniOtDecodeError> for MlKemFerretError {
    fn from(error: MiniOtDecodeError) -> Self {
        Self::MiniOt(error)
    }
}

/// Prover-side state. It holds only the COT receiver pool, not verifier Delta.
pub struct MlKemFerretProver {
    pool: CotPoolReceiver,
}

/// Verifier-side state. It holds the COT sender pool and verifier-only Delta.
pub struct MlKemFerretVerifier {
    pool: CotPoolSender,
    delta: Delta<U1, Galois128>,
}

/// Adapt an application CSPRNG to Volar's deterministic-spec RNG interface.
/// Every protocol random draw still comes from the caller's CryptoRng.
struct CryptoSpecRng<'a>(&'a mut dyn CryptoRng);

impl SpecRng for CryptoSpecRng<'_> {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }
}

fn fresh_block(rng: &mut dyn CryptoRng) -> [u8; 16] {
    let mut block = [0u8; 16];
    for byte in &mut block {
        *byte = rng.next_u32() as u8;
    }
    block
}

fn xor_block(left: &[u8; 16], right: &[u8; 16]) -> [u8; 16] {
    core::array::from_fn(|index| left[index] ^ right[index])
}

fn block_payload(block: &[u8; 16]) -> [u8; SHARED_KEY_BYTES] {
    let mut payload = [0u8; SHARED_KEY_BYTES];
    payload[..16].copy_from_slice(block);
    payload
}

fn to_field_block(block: [u8; 16]) -> Array<Galois128, U1> {
    Array::from_fn(|_| Galois128(u128::from_le_bytes(block)))
}

fn bit_to_field(bit: bool) -> Galois128 {
    Galois128(u128::from(bit))
}

fn encoded_key_batch_len(count: usize) -> usize {
    count * 2 * ENCAPSULATION_KEY_BYTES
}

fn encoded_response_batch_len(count: usize) -> usize {
    count * 2 * (CIPHERTEXT_BYTES + SHARED_KEY_BYTES)
}

fn validate_message_length(bytes: &[u8], expected: usize) -> Result<(), MlKemFerretError> {
    if bytes.len() != expected {
        return Err(MlKemFerretError::MiniOt(MiniOtDecodeError::InvalidLength {
            expected,
            found: bytes.len(),
        }));
    }
    Ok(())
}

fn validate_profiles(
    setup_params: FerretParams,
    main_params: FerretParams,
) -> Result<(usize, usize), MlKemFerretError> {
    let setup_seed_count = checked_seed_cot_count(setup_params)?;
    let main_seed_count = checked_seed_cot_count(main_params)?;
    if setup_params.n > MAX_SETUP_PROFILE_COTS {
        return Err(MlKemFerretError::ProfileTooLarge {
            requested: setup_params.n,
            maximum: MAX_SETUP_PROFILE_COTS,
        });
    }
    if main_params.n > MAX_MAIN_PROFILE_COTS {
        return Err(MlKemFerretError::ProfileTooLarge {
            requested: main_params.n,
            maximum: MAX_MAIN_PROFILE_COTS,
        });
    }
    if main_seed_count > MAX_MAIN_SEED_COTS {
        return Err(MlKemFerretError::MainSeedTooLarge {
            requested: main_seed_count,
            maximum: MAX_MAIN_SEED_COTS,
        });
    }
    if setup_seed_count > MAX_SETUP_SEED_COTS {
        return Err(MlKemFerretError::BootstrapTooLarge {
            requested: setup_seed_count,
            maximum: MAX_SETUP_SEED_COTS,
        });
    }
    if setup_params.n < main_seed_count {
        return Err(MlKemFerretError::SetupOutputTooSmall {
            required: main_seed_count,
            available: setup_params.n,
        });
    }
    Ok((setup_seed_count, main_seed_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use volar_spec::ot::ferret::FERRET_REG_TOY;

    #[test]
    fn parameterized_profiles_are_bounded_before_protocol_io() {
        let oversized_setup = FerretParams {
            n: MAX_SETUP_PROFILE_COTS + 1,
            ..FERRET_REG_SETUP
        };
        assert_eq!(
            validate_profiles(oversized_setup, FERRET_REG_TOY),
            Err(MlKemFerretError::ProfileTooLarge {
                requested: MAX_SETUP_PROFILE_COTS + 1,
                maximum: MAX_SETUP_PROFILE_COTS,
            })
        );

        let oversized_main = FerretParams {
            n: MAX_MAIN_PROFILE_COTS + 1,
            ..FERRET_REG_TOY
        };
        assert_eq!(
            validate_profiles(FERRET_REG_SETUP, oversized_main),
            Err(MlKemFerretError::ProfileTooLarge {
                requested: MAX_MAIN_PROFILE_COTS + 1,
                maximum: MAX_MAIN_PROFILE_COTS,
            })
        );

        let oversized_main_seed = FerretParams {
            n: 1_000_000,
            k: MAX_MAIN_SEED_COTS,
            t: 1,
        };
        assert_eq!(
            validate_profiles(FERRET_REG_SETUP, oversized_main_seed),
            Err(MlKemFerretError::MainSeedTooLarge {
                requested: MAX_MAIN_SEED_COTS + 20,
                maximum: MAX_MAIN_SEED_COTS,
            })
        );
    }
}

/// Establish the paper's Ferret-Reg setup/main profiles using ML-KEM MiniOT
/// directly for the semi-honest setup-seed COTs.
///
/// The standard parameterized entry point is bounded to Ferret-Reg's official
/// setup seed size. It does not make the KEM-to-COT composition standard or
/// production-ready; callers must use a reviewed, authenticated transport.
pub fn mlkem_ferret_prover<Io: StackIo>(
    rng: &mut dyn CryptoRng,
    io: &mut Io,
) -> Result<MlKemFerretProver, MlKemFerretError> {
    mlkem_ferret_prover_with_params(rng, FERRET_REG_SETUP, FERRET_REG_MAIN, io)
}

/// Test/profile variant of [`mlkem_ferret_prover`] with explicit regular
/// setup and main parameters. The seed COT limit is always enforced.
pub fn mlkem_ferret_prover_with_params<Io: StackIo>(
    rng: &mut dyn CryptoRng,
    setup_params: FerretParams,
    main_params: FerretParams,
    io: &mut Io,
) -> Result<MlKemFerretProver, MlKemFerretError> {
    let (m, _) = validate_profiles(setup_params, main_params)?;
    let mut receiver_bits = Vec::with_capacity(m);
    for _ in 0..m {
        receiver_bits.push((rng.next_u32() & 1) != 0);
    }

    let mut receiver_values = Vec::with_capacity(m);
    for batch_start in (0..m).step_by(MINIOT_BATCH_SIZE) {
        let count = (m - batch_start).min(MINIOT_BATCH_SIZE);
        let mut secret_keys = Vec::with_capacity(count);
        let mut encoded_keys = Vec::with_capacity(encoded_key_batch_len(count));
        for choice in &receiver_bits[batch_start..batch_start + count] {
            let (keys, secret_key) = miniot::miniot_start_recv::<2>(rng, usize::from(*choice));
            encoded_keys.extend_from_slice(&miniot::encode_public_keys(&keys));
            secret_keys.push(secret_key);
        }
        io.send(TAG_MLKEM_COT_KEYS, &encoded_keys);

        let responses = io.recv(TAG_MLKEM_COT_RESPONSES);
        validate_message_length(&responses, encoded_response_batch_len(count))?;
        let response_size = 2 * (CIPHERTEXT_BYTES + SHARED_KEY_BYTES);
        for (offset, (choice, secret_key)) in receiver_bits[batch_start..batch_start + count]
            .iter()
            .zip(secret_keys)
            .enumerate()
        {
            let start = offset * response_size;
            let selected = miniot::miniot_finish_recv_encoded::<2>(
                &responses[start..start + response_size],
                usize::from(*choice),
                secret_key,
            )?;
            let mut value = [0u8; 16];
            value.copy_from_slice(&selected[..16]);
            receiver_values.push(value);
        }
    }

    let seed = FerretReceiverSeed {
        u: receiver_bits,
        w: receiver_values,
    };
    let mut setup_pool = CotPoolReceiver::from_seed(setup_params, seed)?;
    {
        let mut ferret_rng = CryptoSpecRng(rng);
        // This is the one-time Ferret-Reg setup iteration, not IKNP.
        stack_refill_receiver(&mut ferret_rng, &mut setup_pool, io);
    }
    let pool = setup_pool.reprofile(main_params)?;
    Ok(MlKemFerretProver { pool })
}

/// Establish the paper's Ferret-Reg setup/main profiles using ML-KEM MiniOT
/// directly for the semi-honest setup-seed COTs.
///
/// The standard parameterized entry point is bounded to Ferret-Reg's official
/// setup seed size. The verifier's `Delta` remains private to this role.
pub fn mlkem_ferret_verifier<Io: StackIo>(
    rng: &mut dyn CryptoRng,
    io: &mut Io,
) -> Result<MlKemFerretVerifier, MlKemFerretError> {
    mlkem_ferret_verifier_with_params(rng, FERRET_REG_SETUP, FERRET_REG_MAIN, io)
}

/// Test/profile variant of [`mlkem_ferret_verifier`] with explicit regular
/// setup and main parameters. The seed COT limit is always enforced.
pub fn mlkem_ferret_verifier_with_params<Io: StackIo>(
    rng: &mut dyn CryptoRng,
    setup_params: FerretParams,
    main_params: FerretParams,
    io: &mut Io,
) -> Result<MlKemFerretVerifier, MlKemFerretError> {
    let (m, _) = validate_profiles(setup_params, main_params)?;
    let delta_msg = (0..64)
        .find_map(|_| {
            let candidate = fresh_block(rng);
            candidate.iter().any(|byte| *byte != 0).then_some(candidate)
        })
        .ok_or(MlKemFerretError::ZeroDelta)?;

    let mut sender_values = Vec::with_capacity(m);
    for batch_start in (0..m).step_by(MINIOT_BATCH_SIZE) {
        let count = (m - batch_start).min(MINIOT_BATCH_SIZE);
        let encoded_keys = io.recv(TAG_MLKEM_COT_KEYS);
        validate_message_length(&encoded_keys, encoded_key_batch_len(count))?;
        let key_pair_size = 2 * ENCAPSULATION_KEY_BYTES;
        let mut responses = Vec::with_capacity(encoded_response_batch_len(count));
        for offset in 0..count {
            let start = offset * key_pair_size;
            let key_pair = &encoded_keys[start..start + key_pair_size];
            let q = fresh_block(rng);
            let q_xor_delta = xor_block(&q, &delta_msg);
            let response = miniot::miniot_sender_encoded::<2>(
                rng,
                [block_payload(&q), block_payload(&q_xor_delta)],
                key_pair,
            )?;
            responses.extend_from_slice(&response);
            sender_values.push(q);
        }
        io.send(TAG_MLKEM_COT_RESPONSES, &responses);
    }

    let seed = FerretSenderSeed {
        delta: delta_msg,
        q: sender_values,
    };
    let mut setup_pool = CotPoolSender::from_seed(setup_params, seed)?;
    {
        let mut ferret_rng = CryptoSpecRng(rng);
        stack_refill_sender(&mut ferret_rng, &mut setup_pool, io);
    }
    let pool = setup_pool.reprofile(main_params)?;
    Ok(MlKemFerretVerifier {
        pool,
        delta: Delta {
            delta: to_field_block(delta_msg),
        },
    })
}

impl MlKemFerretProver {
    /// Convert one fresh/random Ferret COT into this prover's VOLE share.
    pub fn commit_input<Io: StackIo>(
        &mut self,
        rng: &mut dyn CryptoRng,
        io: &mut Io,
        bit: bool,
    ) -> Vope<U1, Galois128, U1> {
        let mut rng = CryptoSpecRng(rng);
        let value = stack_bea95_receiver(&mut rng, &mut self.pool, io, bit);
        vole_commit_bit_prover_share(to_field_block(value), bit_to_field, bit)
    }

    /// Run one fresh Ferret main-profile iteration when the seed watermark is
    /// reached, preserving the remaining COT buffer.
    pub fn refill<Io: StackIo>(&mut self, rng: &mut dyn CryptoRng, io: &mut Io) {
        let mut rng = CryptoSpecRng(rng);
        stack_refill_receiver(&mut rng, &mut self.pool, io);
    }

    /// Number of ready COTs, useful for bounded admission and monitoring.
    pub fn remaining(&self) -> usize {
        self.pool.remaining()
    }
}

impl MlKemFerretVerifier {
    /// Convert one fresh/random Ferret COT into this verifier's VOLE share.
    pub fn commit_input<Io: StackIo>(
        &mut self,
        rng: &mut dyn CryptoRng,
        io: &mut Io,
    ) -> Q<U1, Galois128> {
        let mut rng = CryptoSpecRng(rng);
        let value = stack_bea95_sender(&mut rng, &mut self.pool, io);
        vole_commit_bit_verifier_share(to_field_block(value))
    }

    /// Run one fresh Ferret main-profile iteration when the seed watermark is
    /// reached, preserving the remaining COT buffer.
    pub fn refill<Io: StackIo>(&mut self, rng: &mut dyn CryptoRng, io: &mut Io) {
        let mut rng = CryptoSpecRng(rng);
        stack_refill_sender(&mut rng, &mut self.pool, io);
    }

    /// Verifier-only global correlation used by its gate checks.
    pub fn delta(&self) -> &Delta<U1, Galois128> {
        &self.delta
    }

    /// Number of ready COTs, useful for bounded admission and monitoring.
    pub fn remaining(&self) -> usize {
        self.pool.remaining()
    }
}

/// Execute the verifier context with this session's secret Delta.
pub fn verifier_context<I, H>(
    session: &MlKemFerretVerifier,
    hats: I,
    hook: H,
) -> crate::VoleVerifierContext<U1, Galois128, I, H>
where
    I: Iterator<Item = Array<Galois128, U1>>,
    H: crate::VoleVerifierHook<U1, Galois128>,
{
    crate::VoleVerifierContext {
        delta: session.delta.clone(),
        hats,
        hook,
        gate_index: 0,
    }
}

/// Create the prover execution context. The caller supplies a stream sink for
/// its live `hat` messages; no verifier state is accepted by this interface.
pub fn prover_context<'a, 'b>(
    hats: &'a mut (dyn Pusher<Array<Galois128, U1>> + 'b),
) -> crate::VoleProverContext<'a, 'b, U1, Galois128> {
    crate::VoleProverContext {
        hats,
        bit_to_t: bit_to_field,
    }
}

/// Expose the protocol's message tags to transports and harnesses.
pub mod message_tags {
    /// Prover → verifier: batch of two ML-KEM public keys per setup COT.
    pub const MLKEM_COT_KEYS: u8 = super::TAG_MLKEM_COT_KEYS;
    /// Verifier → prover: batch of two ML-KEM ciphertext/masked-value pairs.
    pub const MLKEM_COT_RESPONSES: u8 = super::TAG_MLKEM_COT_RESPONSES;
    /// Ferret receiver → sender: LPN seed and SPCOT choices.
    pub const FERRET_OPEN: u8 = volar_spec::ot::wire::TAG_FERRET_OPEN;
    /// Ferret sender → receiver: MPCOT ciphertexts.
    pub const FERRET_MPCOT: u8 = volar_spec::ot::wire::TAG_FERRET_MPCOT;
    /// Prover → verifier: Bea95 chosen-bit correction.
    pub const BEA95: u8 = volar_spec::ot::wire::TAG_BEA95;
}
