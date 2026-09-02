//! Truncate Boolar storage addresses so dense [`super::StorageBank`]s fit.
//!
//! WASM/spill lowering produces 32-bit element addresses. Boolar then appends
//! value bit-index bits as the high end of the LSB-first `addr` vector (a
//! 32-bit pointer plus a 2-bit index is 34 wires → `2^34` dense cells).
//!
//! [`trim_storage_element_bits`] drops the high *element* bits and keeps the
//! bit-index suffix. That aliases the address space the same way bounded WASM
//! RAM does (`WaffleImportConfig::with_memory_address_bits`). Full-width
//! addresses without `2^n` allocation are the sparse-banks handoff
//! (`sparse-banks.md`).

use alloc::vec::Vec;
use volar_ir::boolar::BIrStmt;
use volar_ir::circuit::BCircuit;
use volar_ir::ir::IRVarId;

/// WASM linear-memory bound for a dense bank: `2^24` bytes (16 MiB).
///
/// Full WASM pointers are 32-bit (`4 GiB`). This is the trimmed image size
/// used as the runnable dense checkpoint; sparse banks lift it.
pub const WASM_BYTE_ADDRESS_BITS: usize = 24;

/// Why a storage-address trim could not run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddressTrimError {
    /// `element_bits` must not exceed `native_element_bits`.
    ElementWiderThanNative {
        element_bits: usize,
        native_element_bits: usize,
    },
}

/// Drop high element-address bits on every storage access in `circuit`.
///
/// Boolar `addr` layout is LSB-first: bits `[0, native_element_bits)` are the
/// element pointer (WASM/spill: 32); bits at `native_element_bits..` are the
/// value bit-index suffix and are kept. Lanes whose `addr` is no wider than
/// `native_element_bits` are treated as element-only.
///
/// `pre_init` offsets are not remapped. If a segment would fall outside the
/// trimmed dense image, [`super::storage_requirements`] fails closed.
pub fn trim_storage_element_bits<P: Clone>(
    circuit: &mut BCircuit<P>,
    element_bits: usize,
    native_element_bits: usize,
) -> Result<(), AddressTrimError> {
    if element_bits > native_element_bits {
        return Err(AddressTrimError::ElementWiderThanNative {
            element_bits,
            native_element_bits,
        });
    }
    for node in &mut circuit.stmts {
        if let Some(addr) = storage_addr_mut(&mut node.kind) {
            *addr = trim_addr(addr, element_bits, native_element_bits);
        }
    }
    Ok(())
}

/// Drop high bits of every concatenated Boolar storage `addr` until it is at
/// most `max_address_bits` wide (`2^max_address_bits` dense cells).
///
/// Use after [`trim_storage_element_bits`] so a leftover bit-index suffix
/// cannot push the image past the dense cap (e.g. 24-bit element + 6-bit
/// index → 30 bits). [`WASM_BYTE_ADDRESS_BITS`] yields a 16 MiB bool image.
pub fn trim_storage_addr_width<P: Clone>(circuit: &mut BCircuit<P>, max_address_bits: usize) {
    for node in &mut circuit.stmts {
        if let Some(addr) = storage_addr_mut(&mut node.kind) {
            if addr.len() > max_address_bits {
                addr.truncate(max_address_bits);
            }
        }
    }
}

fn storage_addr_mut(stmt: &mut BIrStmt) -> Option<&mut Vec<IRVarId>> {
    match stmt {
        BIrStmt::StorageRead { addr, .. }
        | BIrStmt::StorageWrite { addr, .. }
        | BIrStmt::ActionStoreBit { addr, .. } => Some(addr),
        _ => None,
    }
}

fn trim_addr(addr: &[IRVarId], element_bits: usize, native_element_bits: usize) -> Vec<IRVarId> {
    if addr.len() <= native_element_bits {
        addr.iter().copied().take(element_bits).collect()
    } else {
        let mut out: Vec<IRVarId> = addr.iter().copied().take(element_bits).collect();
        out.extend_from_slice(&addr[native_element_bits..]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use volar_ir::boolar::LaneId;
    use volar_ir_common::{Node, StorageId};

    const STORAGE: StorageId = StorageId(2);
    const LANE: LaneId = LaneId(0);

    fn read_circuit(addr_width: u32) -> BCircuit {
        let addr: Vec<IRVarId> = (0..addr_width).map(IRVarId).collect();
        BCircuit {
            params: addr_width,
            stmts: vec![Node::new(
                BIrStmt::StorageRead {
                    storage: STORAGE,
                    lane: LANE,
                    addr,
                },
                (),
                None,
            )],
            pre_init: vec![],
            outputs: vec![IRVarId(addr_width)],
        }
    }

    fn addr_len(circuit: &BCircuit) -> usize {
        match &circuit.stmts[0].kind {
            BIrStmt::StorageRead { addr, .. } => addr.len(),
            _ => panic!("expected a storage read"),
        }
    }

    #[test]
    fn keeps_bit_index_suffix_when_trimming_a_34_bit_spill_address() {
        let mut circuit = read_circuit(34);
        trim_storage_element_bits(&mut circuit, 5, 32).unwrap();
        assert_eq!(addr_len(&circuit), 7);
        match &circuit.stmts[0].kind {
            BIrStmt::StorageRead { addr, .. } => {
                assert_eq!(addr[0], IRVarId(0));
                assert_eq!(addr[4], IRVarId(4));
                assert_eq!(addr[5], IRVarId(32));
                assert_eq!(addr[6], IRVarId(33));
            }
            _ => panic!("expected a storage read"),
        }
    }

    #[test]
    fn trims_element_only_addrs_from_the_high_end() {
        let mut circuit = read_circuit(8);
        trim_storage_element_bits(&mut circuit, 5, 32).unwrap();
        assert_eq!(addr_len(&circuit), 5);
    }

    #[test]
    fn rejects_element_wider_than_native() {
        let mut circuit = read_circuit(34);
        assert_eq!(
            trim_storage_element_bits(&mut circuit, 33, 32),
            Err(AddressTrimError::ElementWiderThanNative {
                element_bits: 33,
                native_element_bits: 32,
            })
        );
    }

    #[test]
    fn wasm_byte_space_caps_a_34_bit_addr_at_24_bits() {
        let mut circuit = read_circuit(34);
        trim_storage_element_bits(&mut circuit, WASM_BYTE_ADDRESS_BITS, 32).unwrap();
        assert_eq!(addr_len(&circuit), 26);
        trim_storage_addr_width(&mut circuit, WASM_BYTE_ADDRESS_BITS);
        assert_eq!(addr_len(&circuit), WASM_BYTE_ADDRESS_BITS);
    }
}
