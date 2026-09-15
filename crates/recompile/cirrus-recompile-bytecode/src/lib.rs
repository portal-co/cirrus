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
        let entry_len = header.u32()? as usize;
        if !header.is_empty() {
            return Err(DecodeError::InvalidSection);
        }
        let entry = reader.take(entry_len)?;
        if !reader.is_empty() {
            return Err(DecodeError::InvalidSection);
        }
        validate_entry(entry, slots, inputs)?;
        Ok(Self { bytes, slots, inputs, outputs, entry })
    }

    /// The scratch-slot width required by this executable.
    pub const fn slots(self) -> u32 {
        self.slots
    }

    /// The validated canonical executable bytes.
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Execute the program against a caller-owned scratch buffer.
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
            match entry.byte().expect("validated instruction stream") {
                OP_END => break,
                OP_CONST0 | OP_CONST1 => {
                    let out = entry.u32().expect("validated slot") as usize;
                    if scratch[out].is_none() {
                        scratch[out] = Some(cirrus_core::ContextWithCreate::create(backend, entry.last_opcode == OP_CONST1)?);
                    }
                }
                OP_AND | OP_OR | OP_XOR => {
                    let out = entry.u32().expect("validated slot") as usize;
                    let a = entry.u32().expect("validated slot") as usize;
                    let b = entry.u32().expect("validated slot") as usize;
                    let value = match entry.last_opcode {
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

/// Transpile the currently supported flat Boolean `Program` subset.
pub fn transpile(program: &Program) -> Result<Vec<u8>, TranspileError> {
    program.validate().map_err(|_| TranspileError::InvalidProgram)?;
    if !program.externals.is_empty() || !program.storage_ops.is_empty() || !program.storage_banks.is_empty() || !program.storage_init.is_empty() {
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
            Op::External(_) | Op::Storage(_) => return Err(TranspileError::UnsupportedOperation),
        }
    }
    entry.push(OP_END);
    let mut header = Vec::new();
    write_u32(&mut header, slots);
    write_u32(&mut header, u32::try_from(inputs.len()).map_err(|_| TranspileError::TooLarge)?);
    header.extend_from_slice(&inputs);
    write_u32(&mut header, u32::try_from(outputs.len()).map_err(|_| TranspileError::TooLarge)?);
    header.extend_from_slice(&outputs);
    write_u32(&mut header, u32::try_from(entry.len()).map_err(|_| TranspileError::TooLarge)?);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.push(VERSION);
    bytes.push(FLAGS);
    write_u32(&mut bytes, u32::try_from(header.len()).map_err(|_| TranspileError::TooLarge)?);
    bytes.extend_from_slice(&header);
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

fn validate_entry(bytes: &[u8], slots: u32, inputs: &[u8]) -> Result<(), DecodeError> {
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
    last_opcode: u8,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self { Self { bytes, at: 0, last_opcode: 0xff } }
    const fn is_empty(&self) -> bool { self.at == self.bytes.len() }
    fn take(&mut self, count: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.at.checked_add(count).ok_or(DecodeError::Truncated)?;
        let result = self.bytes.get(self.at..end).ok_or(DecodeError::Truncated)?;
        self.at = end;
        Ok(result)
    }
    fn byte(&mut self) -> Result<u8, DecodeError> {
        let byte = *self.take(1)?.first().expect("one requested byte");
        self.last_opcode = byte;
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
