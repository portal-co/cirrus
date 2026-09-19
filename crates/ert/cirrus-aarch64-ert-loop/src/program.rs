//! Optional allocation-backed AArch64 loop program boundary map.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use cirrus_aarch64_ert::{DecodeError, RawMemory};

use crate::{Aarch64IndirectTargets, Aarch64ReturnTargets};

/// A precomputed, validated map of an A64 image's control-flow boundaries.
///
/// This is only a cache for the live body runner: execution remains the same
/// instruction semantics, with the map proving which branch/return PCs have
/// already been discovered.
#[derive(Debug)]
pub struct Aarch64LoopedProgram {
    /// Every conditional branch boundary: pc -> (taken, fallthrough).
    pub(crate) boundaries: BTreeMap<u64, (u64, u64)>,
    /// Every declared indirect `BR`/`BLR` site.
    pub(crate) indirect_sites: Vec<u64>,
    /// Every declared symbolic `RET` site.
    pub(crate) return_sites: Vec<u64>,
}

impl Aarch64LoopedProgram {
    /// Walk the image from `entry` to the audited fixed point.
    ///
    /// Conditional branches enqueue both successors; direct branches follow
    /// the target; declared indirect/return sites enqueue their exhaustive
    /// target sets. SVC exit and canonical concrete `RET` stop a body.
    pub fn compile(
        memory: &RawMemory<'_>,
        entry: u64,
        indirect: &[Aarch64IndirectTargets<'_>],
        returns: &[Aarch64ReturnTargets<'_>],
        max_boundaries: usize,
    ) -> Result<Self, DecodeError> {
        let mut boundaries = BTreeMap::new();
        let mut indirect_sites = Vec::new();
        let mut return_sites = Vec::new();
        let mut visited = BTreeSet::new();
        let mut worklist = alloc::vec![entry];
        while let Some(start) = worklist.pop() {
            if !visited.insert(start) {
                continue;
            }
            let mut pc = start;
            loop {
                let raw =
                    u32::from_le_bytes(memory.read64::<4>(pc).ok_or(DecodeError::Memory(pc))?);
                if raw & 0xfe00_0010 == 0x5400_0000 {
                    if raw & 15 == 14 {
                        pc = pc.wrapping_add(4);
                        continue;
                    }
                    if boundaries.len() >= max_boundaries {
                        return Err(DecodeError::Malformed(raw));
                    }
                    let taken = branch_target(pc, raw);
                    let fallthrough = pc.wrapping_add(4);
                    boundaries.insert(pc, (taken, fallthrough));
                    worklist.push(taken);
                    worklist.push(fallthrough);
                    break;
                }
                if raw & 0x7e00_0000 == 0x3400_0000 || raw & 0x7e00_0000 == 0x3600_0000 {
                    if boundaries.len() >= max_boundaries {
                        return Err(DecodeError::Malformed(raw));
                    }
                    let taken = branch_target(pc, raw);
                    let fallthrough = pc.wrapping_add(4);
                    boundaries.insert(pc, (taken, fallthrough));
                    worklist.push(taken);
                    worklist.push(fallthrough);
                    break;
                }
                if raw & 0x7c00_0000 == 0x1400_0000 {
                    let target = branch_target(pc, raw);
                    worklist.push(target);
                    break;
                }
                if raw & 0xffff_fc1f == 0xd61f_0000 || raw & 0xffff_fc1f == 0xd63f_0000 {
                    let Some(declaration) = indirect.iter().find(|site| site.pc == pc) else {
                        return Err(DecodeError::Unsupported(raw));
                    };
                    if declaration.targets.is_empty() {
                        return Err(DecodeError::Unsupported(raw));
                    }
                    indirect_sites.push(pc);
                    for target in declaration.targets {
                        worklist.push(*target);
                    }
                    if raw & 0xffff_fc1f == 0xd63f_0000 {
                        pc = pc.wrapping_add(4);
                        continue;
                    }
                    break;
                }
                if raw == 0xd65f_03c0 {
                    let Some(declaration) = returns.iter().find(|site| site.pc == pc) else {
                        break;
                    };
                    if declaration.targets.is_empty() {
                        return Err(DecodeError::Unsupported(raw));
                    }
                    return_sites.push(pc);
                    for target in declaration.targets {
                        worklist.push(*target);
                    }
                    break;
                }
                if raw == 0xd400_0001 {
                    break;
                }
                pc = pc.wrapping_add(4);
                if pc >= 4096 {
                    break;
                }
            }
        }
        Ok(Self {
            boundaries,
            indirect_sites,
            return_sites,
        })
    }

    /// The branch-boundary map this program validated.
    pub fn boundaries(&self) -> &BTreeMap<u64, (u64, u64)> {
        &self.boundaries
    }

    /// Every declared indirect branch site.
    pub fn indirect_sites(&self) -> &[u64] {
        &self.indirect_sites
    }

    /// Every declared symbolic return site.
    pub fn return_sites(&self) -> &[u64] {
        &self.return_sites
    }
}

/// Extract the target of an audited direct/conditional A64 branch encoding.
fn branch_target(pc: u64, raw: u32) -> u64 {
    let offset = if raw & 0x7e00_0000 == 0x3600_0000 {
        cirrus_aarch64_ert::test_branch_offset(raw)
    } else if raw & 0x7e00_0000 == 0x3400_0000 || raw & 0x7e00_0010 == 0x5400_0000 {
        cirrus_aarch64_ert::compare_branch_offset(raw)
    } else {
        sign_extend(raw & 0x03ff_ffff, 26) << 2
    };
    pc.wrapping_add_signed(offset)
}

fn sign_extend(value: u32, width: u32) -> i64 {
    (i64::from(value) << (64 - width)) >> (64 - width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_walks_branches_and_declared_returns() {
        // nop; svc #0 (entry at nop)
        let code = [0xd503_201fu32, 0xd400_0001];
        let mut bytes = [0u8; 8];
        for (index, word) in code.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let memory = RawMemory::from_slice(&bytes);
        let program = Aarch64LoopedProgram::compile(&memory, 0, &[], &[], 8).unwrap();
        assert!(program.boundaries().is_empty());
        assert!(program.return_sites().is_empty());
        assert!(program.indirect_sites().is_empty());
    }

    #[test]
    fn compile_rejects_undeclared_indirect_branches() {
        let bytes = 0xd61f_0020u32.to_le_bytes(); // br x1
        assert_eq!(
            Aarch64LoopedProgram::compile(&RawMemory::from_slice(&bytes), 0, &[], &[], 4)
                .unwrap_err(),
            DecodeError::Unsupported(0xd61f_0020)
        );
    }
}
