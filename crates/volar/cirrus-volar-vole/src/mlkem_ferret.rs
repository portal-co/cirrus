//! Role-separated ML-KEM base OT → IKNP COT → refillable Ferret stack.
//!
//! The verifier owns the correlated-OT `Delta`; the prover owns the COT
//! receiver choices/values. Neither session contains the other role's state.
//! The ML-KEM-1024 MiniOT seed bootstrap is semi-honest and is experimental;
//! it is not a malicious-secure protocol or a production-reviewed profile.

use alloc::vec::Vec;
use core::fmt;

use cipher::consts::U1;
use cirrus_core::Pusher;
use hybrid_array::Array;
use miniot::CryptoRng;
use sha3::Sha3_256;
use volar_spec::{
    SpecRng,
    field::Galois128,
    ot::{
        ferret::{
            CotPoolReceiver, CotPoolSender, FerretParams, FerretReceiverSeed, FerretSenderSeed,
            PoolSeedError, checked_seed_cot_count,
        },
        iknp::{
            IKNP_KAPPA, IKNP_KAPPA_BYTES, IknpUMsg, iknp_receiver_finish, iknp_receiver_u_cols,
            iknp_sender_from_u, pack_kappa,
        },
        two_party::{
            StackIo, stack_bea95_receiver, stack_bea95_sender, stack_refill_receiver,
            stack_refill_sender,
        },
        wire::{
            TAG_BEA95, TAG_FERRET_MPCOT, TAG_FERRET_OPEN, TAG_IKNP_CORR, TAG_IKNP_U, encode_iknp_u,
        },
    },
    vole::{
        Delta, Q, Vope,
        setup::{vole_commit_bit_prover_share, vole_commit_bit_verifier_share},
    },
};

const TAG_MLKEM_PUBLIC_KEYS: u8 = 13;
const TAG_MLKEM_CIPHERTEXTS: u8 = 14;

/// Errors detected while establishing the ML-KEM/IKNP/Ferret seed state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlKemFerretError {
    /// Ferret parameters or independently generated seed dimensions are invalid.
    FerretSeed(PoolSeedError),
    /// A fixed-size ML-KEM MiniOT message was malformed.
    MiniOt(miniot::MiniOtDecodeError),
    /// An IKNP message had the wrong shape or contained non-canonical bits.
    InvalidIknpMessage,
    /// Verifier Delta must be nonzero for the Quicksilver verifier relation.
    ZeroDelta,
}

impl fmt::Display for MlKemFerretError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FerretSeed(error) => write!(formatter, "invalid Ferret seed: {error:?}"),
            Self::MiniOt(error) => write!(formatter, "invalid ML-KEM MiniOT message: {error:?}"),
            Self::InvalidIknpMessage => formatter.write_str("invalid IKNP message shape"),
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

impl From<miniot::MiniOtDecodeError> for MlKemFerretError {
    fn from(error: miniot::MiniOtDecodeError) -> Self {
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

fn fresh_seed(rng: &mut dyn CryptoRng) -> [u8; IKNP_KAPPA_BYTES] {
    let mut seed = [0u8; IKNP_KAPPA_BYTES];
    for byte in &mut seed {
        *byte = rng.next_u32() as u8;
    }
    seed
}

fn payload_seed(seed: &[u8; IKNP_KAPPA_BYTES]) -> [u8; 32] {
    let mut payload = [0u8; 32];
    payload[..IKNP_KAPPA_BYTES].copy_from_slice(seed);
    payload
}

fn decode_iknp_u(bytes: &[u8], m: usize) -> Result<IknpUMsg, MlKemFerretError> {
    let Some(header) = bytes.get(..4) else {
        return Err(MlKemFerretError::InvalidIknpMessage);
    };
    let count = u32::from_le_bytes(header.try_into().unwrap()) as usize;
    if count != IKNP_KAPPA {
        return Err(MlKemFerretError::InvalidIknpMessage);
    }
    let col_len = 4usize
        .checked_add(m)
        .ok_or(MlKemFerretError::InvalidIknpMessage)?;
    let expected = 4usize
        .checked_add(
            IKNP_KAPPA
                .checked_mul(col_len)
                .ok_or(MlKemFerretError::InvalidIknpMessage)?,
        )
        .ok_or(MlKemFerretError::InvalidIknpMessage)?;
    if bytes.len() != expected {
        return Err(MlKemFerretError::InvalidIknpMessage);
    }
    let mut offset = 4;
    let mut u_cols = Vec::with_capacity(IKNP_KAPPA);
    for _ in 0..IKNP_KAPPA {
        let encoded_len =
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if encoded_len != m {
            return Err(MlKemFerretError::InvalidIknpMessage);
        }
        let mut column = Vec::with_capacity(m);
        for value in &bytes[offset..offset + m] {
            match value {
                0 => column.push(false),
                1 => column.push(true),
                _ => return Err(MlKemFerretError::InvalidIknpMessage),
            }
        }
        offset += m;
        u_cols.push(column);
    }
    Ok(IknpUMsg { u_cols })
}

fn decode_iknp_corrections(
    bytes: &[u8],
    m: usize,
) -> Result<Vec<[u8; IKNP_KAPPA_BYTES]>, MlKemFerretError> {
    let Some(header) = bytes.get(..4) else {
        return Err(MlKemFerretError::InvalidIknpMessage);
    };
    let count = u32::from_le_bytes(header.try_into().unwrap()) as usize;
    let expected = 4usize
        .checked_add(
            m.checked_mul(IKNP_KAPPA_BYTES)
                .ok_or(MlKemFerretError::InvalidIknpMessage)?,
        )
        .ok_or(MlKemFerretError::InvalidIknpMessage)?;
    if count != m || bytes.len() != expected {
        return Err(MlKemFerretError::InvalidIknpMessage);
    }
    let mut corrections = Vec::with_capacity(m);
    for chunk in bytes[4..].chunks_exact(IKNP_KAPPA_BYTES) {
        let mut row = [0u8; IKNP_KAPPA_BYTES];
        row.copy_from_slice(chunk);
        corrections.push(row);
    }
    Ok(corrections)
}

fn to_field_block(block: [u8; 16]) -> Array<Galois128, U1> {
    Array::from_fn(|_| Galois128(u128::from_le_bytes(block)))
}

fn bit_to_field(bit: bool) -> Galois128 {
    Galois128(u128::from(bit))
}

/// Establish a prover's IKNP/Ferret pool from ML-KEM base OT.
///
/// The caller's `StackIo` must preserve message order and tags. This side
/// receives only its random COT choices/values and never learns verifier Delta.
pub fn mlkem_ferret_prover<Io: StackIo>(
    rng: &mut dyn CryptoRng,
    params: FerretParams,
    io: &mut Io,
) -> Result<MlKemFerretProver, MlKemFerretError> {
    let m = checked_seed_cot_count(params)?;
    let mut receiver_bits = Vec::with_capacity(m);
    for _ in 0..m {
        receiver_bits.push((rng.next_u32() & 1) != 0);
    }

    let mut seeds_0 = [[0u8; IKNP_KAPPA_BYTES]; IKNP_KAPPA];
    let mut seeds_1 = [[0u8; IKNP_KAPPA_BYTES]; IKNP_KAPPA];
    for index in 0..IKNP_KAPPA {
        seeds_0[index] = fresh_seed(rng);
        seeds_1[index] = fresh_seed(rng);
    }

    // The prover is the MiniOT sender/IKNP receiver: the verifier sends a
    // choice-selected decapsulation key and obtains exactly one seed per pair.
    for index in 0..IKNP_KAPPA {
        let public_keys = io.recv(TAG_MLKEM_PUBLIC_KEYS);
        let response = miniot::miniot_sender_encoded::<2>(
            rng,
            [payload_seed(&seeds_0[index]), payload_seed(&seeds_1[index])],
            &public_keys,
        )?;
        io.send(TAG_MLKEM_CIPHERTEXTS, &response);
    }

    let (t_cols, u_msg) = iknp_receiver_u_cols::<Sha3_256>(m, &receiver_bits, &seeds_0, &seeds_1);
    io.send(TAG_IKNP_U, &encode_iknp_u(&u_msg));
    let corrections = decode_iknp_corrections(&io.recv(TAG_IKNP_CORR), m)?;
    let receiver_values =
        iknp_receiver_finish::<Sha3_256, 16>(&receiver_bits, &t_cols, &corrections);

    let seed = FerretReceiverSeed {
        u: receiver_bits,
        w: receiver_values,
    };
    Ok(MlKemFerretProver {
        pool: CotPoolReceiver::from_seed(params, seed)?,
    })
}

/// Establish a verifier's IKNP/Ferret pool from ML-KEM base OT.
///
/// The verifier's Delta never leaves this role. ML-KEM decoy public keys are
/// handcrafted by `miniot` without secret keys; malformed peer keys are
/// parsed once and rejected, never retried/rejection-sampled.
pub fn mlkem_ferret_verifier<Io: StackIo>(
    rng: &mut dyn CryptoRng,
    params: FerretParams,
    io: &mut Io,
) -> Result<MlKemFerretVerifier, MlKemFerretError> {
    let m = checked_seed_cot_count(params)?;
    let delta_msg = (0..64)
        .find_map(|_| {
            let candidate = fresh_block(rng);
            candidate.iter().any(|byte| *byte != 0).then_some(candidate)
        })
        .ok_or(MlKemFerretError::ZeroDelta)?;

    let delta_ot = core::array::from_fn(|_| (rng.next_u32() & 1) != 0);
    let delta_ot_bytes = pack_kappa(&delta_ot);
    let mut chosen_seeds = [[0u8; IKNP_KAPPA_BYTES]; IKNP_KAPPA];

    // The verifier is the MiniOT receiver/IKNP sender and retains each
    // selected decapsulation key until that OT response has been opened.
    for index in 0..IKNP_KAPPA {
        let (public_keys, secret_key) =
            miniot::miniot_start_recv::<2>(rng, usize::from(delta_ot[index]));
        io.send(
            TAG_MLKEM_PUBLIC_KEYS,
            &miniot::encode_public_keys(&public_keys),
        );
        let response = io.recv(TAG_MLKEM_CIPHERTEXTS);
        let selected = miniot::miniot_finish_recv_encoded::<2>(
            &response,
            usize::from(delta_ot[index]),
            secret_key,
        )?;
        chosen_seeds[index].copy_from_slice(&selected[..IKNP_KAPPA_BYTES]);
    }

    let u_msg = decode_iknp_u(&io.recv(TAG_IKNP_U), m)?;
    let (sender_values, corrections) = iknp_sender_from_u::<Sha3_256, 16>(
        m,
        &delta_msg,
        &delta_ot,
        &delta_ot_bytes,
        &chosen_seeds,
        &u_msg,
    );
    io.send(
        TAG_IKNP_CORR,
        &volar_spec::ot::wire::encode_iknp_corr(&corrections),
    );

    let seed = FerretSenderSeed {
        delta: delta_msg,
        q: sender_values,
    };
    Ok(MlKemFerretVerifier {
        pool: CotPoolSender::from_seed(params, seed)?,
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

    /// Run one refillable Ferret iteration, preserving the seed watermark.
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

    /// Run one refillable Ferret iteration, preserving the seed watermark.
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

/// Expose the protocol's post-setup message tags to transports and harnesses.
pub mod message_tags {
    /// Prover → verifier: batch of two ML-KEM public keys for one base OT.
    pub const MLKEM_PUBLIC_KEYS: u8 = super::TAG_MLKEM_PUBLIC_KEYS;
    /// Verifier → prover: ML-KEM ciphertexts and masked base-OT seed payloads.
    pub const MLKEM_CIPHERTEXTS: u8 = super::TAG_MLKEM_CIPHERTEXTS;
    /// Prover → verifier: IKNP `u` columns.
    pub const IKNP_U: u8 = super::TAG_IKNP_U;
    /// Verifier → prover: IKNP correction rows.
    pub const IKNP_CORRECTIONS: u8 = super::TAG_IKNP_CORR;
    /// Ferret receiver → sender: LPN seed and SPCOT choices.
    pub const FERRET_OPEN: u8 = super::TAG_FERRET_OPEN;
    /// Ferret sender → receiver: MPCOT ciphertexts.
    pub const FERRET_MPCOT: u8 = super::TAG_FERRET_MPCOT;
    /// Prover → verifier: Bea95 chosen-bit correction.
    pub const BEA95: u8 = super::TAG_BEA95;
}
