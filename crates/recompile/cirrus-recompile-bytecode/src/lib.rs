#![no_std]
#![warn(missing_docs)]

//! Canonical compact bytecode for validated Boolean Cirrus [`Program`]s.
//!
//! This first implementation deliberately accepts flat Boolean programs only:
//! callers prepare or compact a program elsewhere, and the bytecode writer
//! rejects storage, externals, and loop-only features until their framed forms
//! are implemented. The reader is allocation-free and rejects malformed or
//! noncanonical input before returning an executable view.

extern crate alloc;

/// Generic, capability-gated compact variants, including BinFHE V2 schedules.
pub mod variant;

use alloc::vec::Vec;
use cirrus_recompile_core::{Idx, Op, Program};

const MAGIC: &[u8; 4] = b"CRBC";
const VERSION: u8 = 1;
const FLAGS: u8 = 0;
const OP_END: u8 = 0;
const OP_CONST0: u8 = 1;
const OP_CONST1: u8 = 2;
const OP_AND: u8 = 3;
const OP_OR: u8 = 4;
const OP_XOR: u8 = 5;
const OP_MUX: u8 = 6;
const OP_STORAGE_READ: u8 = 9;
const OP_STORAGE_WRITE: u8 = 10;
const OP_INIT_BITS: u8 = 11;

/// Why a `Program` cannot yet be represented by the compact v1 subset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TranspileError {
    /// The source program itself violates its structural contracts.
    InvalidProgram,
    /// The source uses an operation whose compact record is not implemented.
    UnsupportedOperation,
    /// A count or slot exceeds v1's `u32` representation.
    TooLarge,
}

/// Why compact executable bytes were rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeError {
    /// The byte stream has no `CRBC` magic.
    BadMagic,
    /// The byte stream uses a version this decoder does not implement.
    UnsupportedVersion,
    /// Reserved flags were set.
    UnsupportedFlags,
    /// A length or record extends past the provided byte slice.
    Truncated,
    /// A LEB128 integer is overlong, noncanonical, or overflows `u32`.
    InvalidInteger,
    /// The fixed v1 section layout is malformed.
    InvalidSection,
    /// The instruction range has an unknown/malformed record or lacks its end.
    InvalidInstruction,
    /// A slot reference is out of range or violates SSA order.
    InvalidSlot,
}

/// A validated, borrowed compact executable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactProgram<'a> {
    bytes: &'a [u8],
    slots: u32,
    inputs: &'a [u8],
    outputs: &'a [u8],
    banks: &'a [u8],
    init: &'a [u8],
    entry: &'a [u8],
}

impl<'a> CompactProgram<'a> {
    /// Validate one compact byte stream without allocating decoded records.
    pub fn validate(bytes: &'a [u8]) -> Result<Self, DecodeError> {
        let mut reader = Reader::new(bytes);
        if reader.take(4)? != MAGIC {
            return Err(DecodeError::BadMagic);
        }
        if reader.byte()? != VERSION {
            return Err(DecodeError::UnsupportedVersion);
        }
        if reader.byte()? != FLAGS {
            return Err(DecodeError::UnsupportedFlags);
        }
        let header_len = reader.u32()? as usize;
        let header = reader.take(header_len)?;
        let mut header = Reader::new(header);
        let slots = header.u32()?;
        let inputs_len = header.u32()? as usize;
        let inputs = header.take(inputs_len)?;
        validate_slot_table(inputs, slots)?;
        let outputs_len = header.u32()? as usize;
        let outputs = header.take(outputs_len)?;
        validate_slot_table(outputs, slots)?;
        let banks_len = header.u32()? as usize;
        let init_len = header.u32()? as usize;
        let entry_len = header.u32()? as usize;
        if !header.is_empty() {
            return Err(DecodeError::InvalidSection);
        }
        let banks = reader.take(banks_len)?;
        let bank_count = validate_banks(banks)?;
        let init = reader.take(init_len)?;
        validate_init(init, banks, bank_count)?;
        let entry = reader.take(entry_len)?;
        if !reader.is_empty() {
            return Err(DecodeError::InvalidSection);
        }
        validate_entry(entry, slots, inputs, banks, bank_count)?;
        Ok(Self { bytes, slots, inputs, outputs, banks, init, entry })
    }

    /// The scratch-slot width required by this executable.
    pub const fn slots(self) -> u32 {
        self.slots
    }

    /// The validated canonical executable bytes.
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Execute a program with no storage records against a caller-owned scratch buffer.
    pub fn execute<Backend>(
        self,
        backend: &mut Backend,
        scratch: &mut [Option<Backend::Wrapped>],
        inputs: &[Backend::Wrapped],
    ) -> Result<Vec<Backend::Wrapped>, Backend::Error>
    where
        Backend: cirrus_core::ContextWithBitAnd<bool>
            + cirrus_core::ContextWithBitOr<bool>
            + cirrus_core::ContextWithBitXor<bool>
            + cirrus_core::ContextWithCreate<bool>
            + cirrus_core::ContextWithMux<bool>,
        Backend::Wrapped: Clone,
    {
        assert!(self.banks.is_empty() && self.init.is_empty(), "storage executable needs execute_with_storage");
        assert_eq!(scratch.len(), self.slots as usize, "scratch width must match compact program");
        let input_count = count_slot_table(self.inputs).expect("validated input table");
        assert_eq!(inputs.len(), input_count, "input count must match compact program");
        for slot in scratch.iter_mut() {
            *slot = None;
        }
        let mut input_reader = Reader::new(self.inputs);
        for value in inputs {
            let slot = input_reader.u32().expect("validated input table") as usize;
            scratch[slot] = Some(value.clone());
        }
        let mut entry = Reader::new(self.entry);
        loop {
            let opcode = entry.byte().expect("validated instruction stream");
            match opcode {
                OP_END => break,
                OP_CONST0 | OP_CONST1 => {
                    let out = entry.u32().expect("validated slot") as usize;
                    if scratch[out].is_none() {
                        scratch[out] = Some(cirrus_core::ContextWithCreate::create(backend, opcode == OP_CONST1)?);
                    }
                }
                OP_AND | OP_OR | OP_XOR => {
                    let out = entry.u32().expect("validated slot") as usize;
                    let a = entry.u32().expect("validated slot") as usize;
                    let b = entry.u32().expect("validated slot") as usize;
                    let value = match opcode {
                        OP_AND => cirrus_core::ContextWithBitAnd::bitand(backend, scratch[a].clone().unwrap(), scratch[b].clone().unwrap())?,
                        OP_OR => cirrus_core::ContextWithBitOr::bitor(backend, scratch[a].clone().unwrap(), scratch[b].clone().unwrap())?,
                        _ => cirrus_core::ContextWithBitXor::bitxor(backend, scratch[a].clone().unwrap(), scratch[b].clone().unwrap())?,
                    };
                    scratch[out] = Some(value);
                }
                OP_MUX => {
                    let out = entry.u32().expect("validated slot") as usize;
                    let cond = entry.u32().expect("validated slot") as usize;
                    let then = entry.u32().expect("validated slot") as usize;
                    let r#else = entry.u32().expect("validated slot") as usize;
                    scratch[out] = Some(cirrus_core::ContextWithMux::mux(
                        backend, scratch[cond].clone().unwrap(), scratch[then].clone().unwrap(), scratch[r#else].clone().unwrap(),
                    )?);
                }
                OP_STORAGE_READ | OP_STORAGE_WRITE => unreachable!("storage executable needs execute_with_storage"),
                _ => unreachable!("validated compact instruction"),
            }
        }
        let mut outputs = Reader::new(self.outputs);
        let mut result = Vec::with_capacity(count_slot_table(self.outputs).expect("validated output table"));
        while !outputs.is_empty() {
            result.push(scratch[outputs.u32().expect("validated output table") as usize].clone().unwrap());
        }
        Ok(result)
    }
}

/// One caller-owned storage bank available to a compact executable.
pub struct RuntimeStorageBank<'a, S: ?Sized> {
    /// Canonical logical storage namespace from the compact bank table.
    pub storage: u32,
    /// Canonical one-bit lane from the compact bank table.
    pub lane: u32,
    /// Exact least-significant-first address width.
    pub address_bits: u32,
    /// Backend-owned storage value.
    pub value: &'a mut S,
}

impl<'a> CompactProgram<'a> {
    /// Apply the validated static storage initialization range once.
    pub fn initialize_storage<Backend>(
        self,
        backend: &mut Backend,
        banks: &mut [RuntimeStorageBank<'_, Backend::Storage>],
    ) -> Result<(), Backend::Error>
    where
        Backend: cirrus_core::ContextWithCreate<bool> + cirrus_core::ContextWithStorage<bool>,
        Backend::Wrapped: Clone,
    {
        validate_runtime_banks(self.banks, banks);
        let mut zero = None;
        let mut one = None;
        let mut init = Reader::new(self.init);
        while !init.is_empty() {
            debug_assert_eq!(init.byte().expect("validated init"), OP_INIT_BITS);
            let bank = init.u32().expect("validated init");
            let mut address = init.u32().expect("validated init");
            let count = init.u32().expect("validated init");
            let (_, _, address_bits) = bank_metadata(self.banks, bank).expect("validated bank");
            let storage = runtime_bank(banks, self.banks, bank);
            for offset in 0..count {
                let bit = init.byte().expect("validated init") != 0;
                let wire = if bit {
                    one.get_or_insert(cirrus_core::ContextWithCreate::create(backend, true)?).clone()
                } else {
                    zero.get_or_insert(cirrus_core::ContextWithCreate::create(backend, false)?).clone()
                };
                let address_bits = (0..address_bits)
                    .map(|bit| cirrus_core::StorageAddressBit {
                        wire: wire.clone(),
                        known: Some((address >> bit) & 1 != 0),
                    })
                    .collect::<Vec<_>>();
                backend.storage_write(storage, &address_bits, wire)?;
                if offset + 1 < count { address += 1; }
            }
        }
        Ok(())
    }

    /// Execute a compact program and apply static initialization first.
    pub fn execute_with_storage<Backend>(
        self,
        backend: &mut Backend,
        scratch: &mut [Option<Backend::Wrapped>],
        inputs: &[Backend::Wrapped],
        banks: &mut [RuntimeStorageBank<'_, Backend::Storage>],
    ) -> Result<Vec<Backend::Wrapped>, Backend::Error>
    where
        Backend: cirrus_core::ContextWithBitAnd<bool>
            + cirrus_core::ContextWithBitOr<bool>
            + cirrus_core::ContextWithBitXor<bool>
            + cirrus_core::ContextWithCreate<bool>
            + cirrus_core::ContextWithMux<bool>
            + cirrus_core::ContextWithStorage<bool>,
        Backend::Wrapped: Clone,
    {
        self.initialize_storage(backend, banks)?;
        self.execute_storage_initialized(backend, scratch, inputs, banks)
    }

    /// Execute a compact program against storage initialized earlier.
    pub fn execute_storage_initialized<Backend>(
        self,
        backend: &mut Backend,
        scratch: &mut [Option<Backend::Wrapped>],
        inputs: &[Backend::Wrapped],
        banks: &mut [RuntimeStorageBank<'_, Backend::Storage>],
    ) -> Result<Vec<Backend::Wrapped>, Backend::Error>
    where
        Backend: cirrus_core::ContextWithBitAnd<bool>
            + cirrus_core::ContextWithBitOr<bool>
            + cirrus_core::ContextWithBitXor<bool>
            + cirrus_core::ContextWithCreate<bool>
            + cirrus_core::ContextWithMux<bool>
            + cirrus_core::ContextWithStorage<bool>,
        Backend::Wrapped: Clone,
    {
        assert_eq!(scratch.len(), self.slots as usize, "scratch width must match compact program");
        validate_runtime_banks(self.banks, banks);
        for slot in scratch.iter_mut() { *slot = None; }
        let input_count = count_slot_table(self.inputs).expect("validated input table");
        assert_eq!(inputs.len(), input_count, "input count must match compact program");
        let mut facts = alloc::vec![None; self.slots as usize];
        let mut input_table = Reader::new(self.inputs);
        for value in inputs {
            scratch[input_table.u32().expect("validated input") as usize] = Some(value.clone());
        }
        let mut entry = Reader::new(self.entry);
        loop {
            let opcode = entry.byte().expect("validated instruction stream");
            match opcode {
                OP_END => break,
                OP_CONST0 | OP_CONST1 => {
                    let out = entry.u32().expect("validated slot") as usize;
                    let value = opcode == OP_CONST1;
                    facts[out] = Some(value);
                    if scratch[out].is_none() {
                        scratch[out] = Some(cirrus_core::ContextWithCreate::create(backend, value)?);
                    }
                }
                OP_AND | OP_OR | OP_XOR => {
                    let out = entry.u32().expect("validated slot") as usize;
                    let a = entry.u32().expect("validated slot") as usize;
                    let b = entry.u32().expect("validated slot") as usize;
                    facts[out] = facts[a].zip(facts[b]).map(|(a, b)| match opcode {
                        OP_AND => a & b, OP_OR => a | b, _ => a ^ b,
                    });
                    scratch[out] = Some(match opcode {
                        OP_AND => cirrus_core::ContextWithBitAnd::bitand(backend, scratch[a].clone().unwrap(), scratch[b].clone().unwrap())?,
                        OP_OR => cirrus_core::ContextWithBitOr::bitor(backend, scratch[a].clone().unwrap(), scratch[b].clone().unwrap())?,
                        _ => cirrus_core::ContextWithBitXor::bitxor(backend, scratch[a].clone().unwrap(), scratch[b].clone().unwrap())?,
                    });
                }
                OP_MUX => {
                    let out = entry.u32().expect("validated slot") as usize;
                    let cond = entry.u32().expect("validated slot") as usize;
                    let then = entry.u32().expect("validated slot") as usize;
                    let r#else = entry.u32().expect("validated slot") as usize;
                    facts[out] = facts[cond].and_then(|cond| if cond { facts[then] } else { facts[r#else] });
                    scratch[out] = Some(cirrus_core::ContextWithMux::mux(
                        backend, scratch[cond].clone().unwrap(), scratch[then].clone().unwrap(), scratch[r#else].clone().unwrap(),
                    )?);
                }
                OP_STORAGE_READ | OP_STORAGE_WRITE => {
                    let read = opcode == OP_STORAGE_READ;
                    let out = read.then(|| entry.u32().expect("validated slot") as usize);
                    let bank = entry.u32().expect("validated bank");
                    let count = entry.u32().expect("validated address count");
                    let address = (0..count).map(|_| {
                        let slot = entry.u32().expect("validated address slot") as usize;
                        cirrus_core::StorageAddressBit { wire: scratch[slot].clone().unwrap(), known: facts[slot] }
                    }).collect::<Vec<_>>();
                    let storage = runtime_bank(banks, self.banks, bank);
                    if let Some(out) = out {
                        scratch[out] = Some(backend.storage_read(storage, &address)?);
                    } else {
                        let value = entry.u32().expect("validated value slot") as usize;
                        backend.storage_write(storage, &address, scratch[value].clone().unwrap())?;
                    }
                }
                _ => unreachable!("validated compact instruction"),
            }
        }
        let mut outputs = Reader::new(self.outputs);
        let mut result = Vec::with_capacity(count_slot_table(self.outputs).expect("validated output table"));
        while !outputs.is_empty() {
            result.push(scratch[outputs.u32().expect("validated output table") as usize].clone().unwrap());
        }
        Ok(result)
    }
}

fn validate_runtime_banks<S: ?Sized>(bytes: &[u8], banks: &[RuntimeStorageBank<'_, S>]) {
    let mut reader = Reader::new(bytes);
    while !reader.is_empty() {
        let storage = reader.u32().expect("validated bank table");
        let lane = reader.u32().expect("validated bank table");
        let address_bits = reader.u32().expect("validated bank table");
        assert_eq!(banks.iter().filter(|bank| bank.storage == storage && bank.lane == lane && bank.address_bits == address_bits).count(), 1, "every compact bank needs one runtime bank");
    }
}

fn bank_metadata(bytes: &[u8], index: u32) -> Result<(u32, u32, u32), DecodeError> {
    let mut reader = Reader::new(bytes);
    for current in 0..=index {
        let storage = reader.u32()?;
        let lane = reader.u32()?;
        let address_bits = reader.u32()?;
        if current == index { return Ok((storage, lane, address_bits)); }
    }
    Err(DecodeError::InvalidSection)
}

fn runtime_bank<'a, S: ?Sized>(
    banks: &'a mut [RuntimeStorageBank<'_, S>], bytes: &[u8], index: u32,
) -> &'a mut S {
    let (storage, lane, address_bits) = bank_metadata(bytes, index).expect("validated bank");
    banks.iter_mut().find(|bank| bank.storage == storage && bank.lane == lane && bank.address_bits == address_bits).expect("validated runtime bank").value
}

/// Transpile the currently supported flat Boolean `Program` subset.
pub fn transpile(program: &Program) -> Result<Vec<u8>, TranspileError> {
    program.validate().map_err(|_| TranspileError::InvalidProgram)?;
    if !program.externals.is_empty() {
        return Err(TranspileError::UnsupportedOperation);
    }
    let slots = u32::try_from(program.ops.len()).map_err(|_| TranspileError::TooLarge)?;
    let mut inputs = Vec::new();
    write_slot_table(&mut inputs, &program.inputs)?;
    let mut outputs = Vec::new();
    write_slot_table(&mut outputs, &program.outputs)?;
    let mut entry = Vec::new();
    for (index, op) in program.ops.iter().enumerate() {
        let out = Idx(index as u32);
        match *op {
            Op::Create(value) => {
                entry.push(if value { OP_CONST1 } else { OP_CONST0 });
                write_u32(&mut entry, out.0);
            }
            Op::BitAnd(a, b) => write_binary(&mut entry, OP_AND, out, a, b),
            Op::BitOr(a, b) => write_binary(&mut entry, OP_OR, out, a, b),
            Op::BitXor(a, b) => write_binary(&mut entry, OP_XOR, out, a, b),
            Op::Mux { cond, then, r#else } => {
                entry.push(OP_MUX);
                for slot in [out, cond, then, r#else] { write_u32(&mut entry, slot.0); }
            }
            Op::External(_) => return Err(TranspileError::UnsupportedOperation),
            Op::Storage(id) => {
                let storage = program.storage_ops.get(id as usize).ok_or(TranspileError::InvalidProgram)?;
                entry.push(match storage.kind {
                    cirrus_recompile_core::StorageOpKind::Read => OP_STORAGE_READ,
                    cirrus_recompile_core::StorageOpKind::Write => OP_STORAGE_WRITE,
                });
                if storage.kind == cirrus_recompile_core::StorageOpKind::Read {
                    write_u32(&mut entry, out.0);
                }
                write_u32(&mut entry, storage.bank);
                write_u32(&mut entry, u32::try_from(storage.address.len()).map_err(|_| TranspileError::TooLarge)?);
                for slot in &storage.address { write_u32(&mut entry, slot.0); }
                if let Some(value) = storage.value { write_u32(&mut entry, value.0); }
            }
        }
    }
    entry.push(OP_END);
    let mut header = Vec::new();
    write_u32(&mut header, slots);
    write_u32(&mut header, u32::try_from(inputs.len()).map_err(|_| TranspileError::TooLarge)?);
    header.extend_from_slice(&inputs);
    write_u32(&mut header, u32::try_from(outputs.len()).map_err(|_| TranspileError::TooLarge)?);
    header.extend_from_slice(&outputs);
    let mut banks = Vec::new();
    for bank in &program.storage_banks {
        write_u32(&mut banks, u32::try_from(bank.storage).map_err(|_| TranspileError::TooLarge)?);
        write_u32(&mut banks, u32::try_from(bank.lane).map_err(|_| TranspileError::TooLarge)?);
        write_u32(&mut banks, bank.address_bits);
    }
    let mut init = Vec::new();
    for segment in &program.storage_init {
        init.push(OP_INIT_BITS);
        write_u32(&mut init, segment.bank);
        let mut address = 0u32;
        for (bit, value) in segment.addr.iter().copied().enumerate() {
            if value { address |= 1u32.checked_shl(bit as u32).ok_or(TranspileError::TooLarge)?; }
        }
        write_u32(&mut init, address);
        write_u32(&mut init, u32::try_from(segment.data.len()).map_err(|_| TranspileError::TooLarge)?);
        for bit in &segment.data { init.push(u8::from(*bit)); }
    }
    write_u32(&mut header, u32::try_from(banks.len()).map_err(|_| TranspileError::TooLarge)?);
    write_u32(&mut header, u32::try_from(init.len()).map_err(|_| TranspileError::TooLarge)?);
    write_u32(&mut header, u32::try_from(entry.len()).map_err(|_| TranspileError::TooLarge)?);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.push(VERSION);
    bytes.push(FLAGS);
    write_u32(&mut bytes, u32::try_from(header.len()).map_err(|_| TranspileError::TooLarge)?);
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&banks);
    bytes.extend_from_slice(&init);
    bytes.extend_from_slice(&entry);
    Ok(bytes)
}

fn write_binary(bytes: &mut Vec<u8>, opcode: u8, out: Idx, a: Idx, b: Idx) {
    bytes.push(opcode);
    for slot in [out, a, b] { write_u32(bytes, slot.0); }
}

fn write_slot_table(bytes: &mut Vec<u8>, slots: &[Idx]) -> Result<(), TranspileError> {
    for slot in slots { write_u32(bytes, slot.0); }
    Ok(())
}

fn write_u32(bytes: &mut Vec<u8>, mut value: u32) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 { byte |= 0x80; }
        bytes.push(byte);
        if value == 0 { return; }
    }
}

fn validate_slot_table(bytes: &[u8], slots: u32) -> Result<(), DecodeError> {
    let mut reader = Reader::new(bytes);
    while !reader.is_empty() {
        if reader.u32()? >= slots { return Err(DecodeError::InvalidSlot); }
    }
    Ok(())
}

fn count_slot_table(bytes: &[u8]) -> Result<usize, DecodeError> {
    let mut reader = Reader::new(bytes);
    let mut count = 0;
    while !reader.is_empty() { reader.u32()?; count += 1; }
    Ok(count)
}

fn validate_banks(bytes: &[u8]) -> Result<usize, DecodeError> {
    let mut reader = Reader::new(bytes);
    let mut count = 0;
    let mut previous = None;
    while !reader.is_empty() {
        let storage = reader.u32()?;
        let lane = reader.u32()?;
        let address_bits = reader.u32()?;
        if address_bits > 32 || previous.is_some_and(|previous| previous >= (storage, lane)) {
            return Err(DecodeError::InvalidSection);
        }
        previous = Some((storage, lane));
        count += 1;
    }
    Ok(count)
}

fn bank_address_bits(bytes: &[u8], index: u32) -> Result<u32, DecodeError> {
    let mut reader = Reader::new(bytes);
    for current in 0..=index {
        reader.u32()?;
        reader.u32()?;
        let address_bits = reader.u32()?;
        if current == index { return Ok(address_bits); }
    }
    Err(DecodeError::InvalidSection)
}

fn validate_init(bytes: &[u8], banks: &[u8], bank_count: usize) -> Result<(), DecodeError> {
    let mut reader = Reader::new(bytes);
    while !reader.is_empty() {
        if reader.byte()? != OP_INIT_BITS { return Err(DecodeError::InvalidInstruction); }
        let bank = reader.u32()?;
        if bank as usize >= bank_count { return Err(DecodeError::InvalidSection); }
        let address = reader.u32()?;
        let bits = reader.u32()?;
        let address_bits = bank_address_bits(banks, bank)?;
        let capacity = 1u64 << address_bits;
        if bits == 0 || u64::from(address) >= capacity || u64::from(address) + u64::from(bits) > capacity {
            return Err(DecodeError::InvalidSection);
        }
        for _ in 0..bits {
            if reader.byte()? > 1 { return Err(DecodeError::InvalidInstruction); }
        }
    }
    Ok(())
}

fn validate_entry(bytes: &[u8], slots: u32, inputs: &[u8], banks: &[u8], bank_count: usize) -> Result<(), DecodeError> {
    let mut input = Reader::new(inputs);
    let mut defined = alloc::vec![false; slots as usize];
    while !input.is_empty() { defined[input.u32()? as usize] = true; }
    let mut reader = Reader::new(bytes);
    loop {
        let opcode = reader.byte()?;
        let operands = match opcode {
            OP_END => {
                if !reader.is_empty() { return Err(DecodeError::InvalidInstruction); }
                return Ok(());
            }
            OP_CONST0 | OP_CONST1 => 1,
            OP_AND | OP_OR | OP_XOR => 3,
            OP_MUX => 4,
            OP_STORAGE_READ | OP_STORAGE_WRITE => {
                let has_out = opcode == OP_STORAGE_READ;
                let out = if has_out { Some(reader.u32()?) } else { None };
                let bank = reader.u32()?;
                if bank as usize >= bank_count { return Err(DecodeError::InvalidSection); }
                let count = reader.u32()?;
                if count != bank_address_bits(banks, bank)? { return Err(DecodeError::InvalidSection); }
                if let Some(out) = out {
                    if out >= slots || defined[out as usize] { return Err(DecodeError::InvalidSlot); }
                }
                for _ in 0..count {
                    let slot = reader.u32()?;
                    if slot >= slots || !defined[slot as usize] { return Err(DecodeError::InvalidSlot); }
                }
                if opcode == OP_STORAGE_WRITE {
                    let value = reader.u32()?;
                    if value >= slots || !defined[value as usize] { return Err(DecodeError::InvalidSlot); }
                }
                if let Some(out) = out { defined[out as usize] = true; }
                continue;
            }
            _ => return Err(DecodeError::InvalidInstruction),
        };
        let out = reader.u32()?;
        if out >= slots { return Err(DecodeError::InvalidSlot); }
        if defined[out as usize] && !matches!(opcode, OP_CONST0 | OP_CONST1) { return Err(DecodeError::InvalidSlot); }
        for _ in 1..operands {
            let slot = reader.u32()?;
            if slot >= out || !defined[slot as usize] { return Err(DecodeError::InvalidSlot); }
        }
        defined[out as usize] = true;
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self { Self { bytes, at: 0 } }
    const fn is_empty(&self) -> bool { self.at == self.bytes.len() }
    fn take(&mut self, count: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.at.checked_add(count).ok_or(DecodeError::Truncated)?;
        let result = self.bytes.get(self.at..end).ok_or(DecodeError::Truncated)?;
        self.at = end;
        Ok(result)
    }
    fn byte(&mut self) -> Result<u8, DecodeError> {
        let byte = *self.take(1)?.first().expect("one requested byte");
        Ok(byte)
    }
    fn u32(&mut self) -> Result<u32, DecodeError> {
        let mut value = 0u32;
        for byte_index in 0..5 {
            let byte = *self.take(1)?.first().expect("one requested byte");
            let payload = u32::from(byte & 0x7f);
            if byte_index == 4 && (payload > 0x0f || byte & 0x80 != 0) { return Err(DecodeError::InvalidInteger); }
            value |= payload << (byte_index * 7);
            if byte & 0x80 == 0 {
                if byte_index > 0 && payload == 0 { return Err(DecodeError::InvalidInteger); }
                return Ok(value);
            }
        }
        Err(DecodeError::InvalidInteger)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_round_trip_executes_boolean_program() {
        let program = Program {
            ops: alloc::vec![Op::Create(true), Op::Create(false), Op::BitXor(Idx(0), Idx(1))],
            inputs: alloc::vec![],
            outputs: alloc::vec![Idx(2)],
            externals: alloc::vec![],
            storage_ops: alloc::vec![],
            storage_banks: alloc::vec![],
            storage_init: alloc::vec![],
        };
        let bytes = transpile(&program).unwrap();
        let compact = CompactProgram::validate(&bytes).unwrap();
        assert_eq!(compact.execute(&mut (), &mut [None, None, None], &[]), Ok(alloc::vec![true]));
        assert_eq!(transpile(&program).unwrap(), bytes);
    }

    #[test]
    fn frames_storage_and_static_initialization() {
        let program = Program {
            ops: alloc::vec![Op::Create(false), Op::Create(true), Op::Storage(0), Op::Storage(1)],
            inputs: alloc::vec![],
            outputs: alloc::vec![Idx(3)],
            externals: alloc::vec![],
            storage_ops: alloc::vec![
                cirrus_recompile_core::StorageOp {
                    kind: cirrus_recompile_core::StorageOpKind::Write,
                    bank: 0,
                    address: alloc::vec![Idx(0)],
                    value: Some(Idx(1)),
                },
                cirrus_recompile_core::StorageOp {
                    kind: cirrus_recompile_core::StorageOpKind::Read,
                    bank: 0,
                    address: alloc::vec![Idx(0)],
                    value: None,
                },
            ],
            storage_banks: alloc::vec![cirrus_recompile_core::StorageBank {
                storage: 9,
                lane: 2,
                address_bits: 1,
            }],
            storage_init: alloc::vec![cirrus_recompile_core::StorageInitSegment {
                bank: 0,
                addr: alloc::vec![false],
                data: alloc::vec![true],
            }],
        };
        let bytes = transpile(&program).unwrap();
        let compact = CompactProgram::validate(&bytes).unwrap();
        assert_eq!(compact.slots(), 4);
        assert!(!compact.banks.is_empty());
        assert!(!compact.init.is_empty());
        let mut cells = [false; 2];
        let mut banks = [RuntimeStorageBank {
            storage: 9,
            lane: 2,
            address_bits: 1,
            value: &mut cells[..],
        }];
        assert_eq!(
            compact.execute_with_storage(&mut (), &mut [None, None, None, None], &[], &mut banks),
            Ok(alloc::vec![true])
        );
        assert_eq!(banks[0].value, [true, true]);
    }

    #[test]
    fn rejects_noncanonical_integer() {
        let mut bytes = transpile(&Program {
            ops: alloc::vec![], inputs: alloc::vec![], outputs: alloc::vec![], externals: alloc::vec![],
            storage_ops: alloc::vec![], storage_banks: alloc::vec![], storage_init: alloc::vec![],
        }).unwrap();
        bytes[6] = 0x80;
        bytes.insert(7, 0x00);
        assert_eq!(CompactProgram::validate(&bytes), Err(DecodeError::InvalidInteger));
    }
}
