#![warn(missing_docs)]

//! Pinned runtime functions for the `bool` scratch-buffer source type.
//!
//! Every code-generating backend (assembly, LLVM, Rust) lowers a
//! `cirrus_recompile_core::Program` into a flat sequence of calls to exactly
//! these functions, by exact symbol name, over indices into a caller-owned
//! scratch buffer of `program.ops.len()` bytes (one byte per bit -- 0 or 1).
//! None of these functions allocate, branch on unknown data, or fail: a
//! [`Program`](cirrus_recompile_core::Program)'s indices are always valid for
//! the buffer a caller sized to match it, so there is nothing left to check
//! at this layer.
//!
//! This is also the documented extension point for later source types (for
//! example a Ring-LWE label buffer): a future `cirrus-recompile-rt-<type>`
//! crate would give these same five symbols bodies that call back into
//! `cirrus-core`'s real `ContextWithBitAnd`/etc. to do actual garbling-table
//! generation, while every backend's *emitted code* is unchanged -- it never
//! knew the buffer held `bool`s instead of labels, only that it called
//! functions with these names and this shape.

use cirrus_recompile_core::{Idx, Op, Program};

/// Materialize a known Boolean constant into `buf[out]`.
///
/// # Safety
///
/// `buf` must be valid for a byte write at `out`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cirrus_rt_create(buf: *mut u8, val: u8, out: u32) {
    unsafe { *buf.add(out as usize) = val & 1 };
}

/// `buf[out] = buf[a] & buf[b]`.
///
/// # Safety
///
/// `buf` must be valid for byte reads at `a`, `b` and a byte write at `out`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cirrus_rt_bitand(buf: *mut u8, a: u32, b: u32, out: u32) {
    unsafe { *buf.add(out as usize) = *buf.add(a as usize) & *buf.add(b as usize) };
}

/// `buf[out] = buf[a] | buf[b]`.
///
/// # Safety
///
/// `buf` must be valid for byte reads at `a`, `b` and a byte write at `out`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cirrus_rt_bitor(buf: *mut u8, a: u32, b: u32, out: u32) {
    unsafe { *buf.add(out as usize) = *buf.add(a as usize) | *buf.add(b as usize) };
}

/// `buf[out] = buf[a] ^ buf[b]`.
///
/// # Safety
///
/// `buf` must be valid for byte reads at `a`, `b` and a byte write at `out`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cirrus_rt_bitxor(buf: *mut u8, a: u32, b: u32, out: u32) {
    unsafe { *buf.add(out as usize) = *buf.add(a as usize) ^ *buf.add(b as usize) };
}

/// `buf[out] = if buf[cond] != 0 { buf[then] } else { buf[r#else] }`.
///
/// # Safety
///
/// `buf` must be valid for byte reads at `cond`, `then`, `r#else` and a byte
/// write at `out`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cirrus_rt_mux(buf: *mut u8, cond: u32, then: u32, r#else: u32, out: u32) {
    unsafe {
        *buf.add(out as usize) = if *buf.add(cond as usize) != 0 {
            *buf.add(then as usize)
        } else {
            *buf.add(r#else as usize)
        };
    }
}

/// Run a whole [`Program`] directly against these pinned functions.
///
/// This is the reference "does the pinned ABI itself preserve semantics"
/// check: every backend's execution test compares its own lowering's output
/// against this same buffer-and-symbol contract, so a mismatch localizes to
/// the backend rather than to a misunderstanding of the ABI.
pub fn execute(program: &Program, inputs: &[bool]) -> Vec<bool> {
    assert_eq!(
        inputs.len(),
        program.inputs.len(),
        "input count must match the recorded program's input slots"
    );
    let mut buf = vec![0u8; program.ops.len()];
    for (&idx, &value) in program.inputs.iter().zip(inputs) {
        unsafe { cirrus_rt_create(buf.as_mut_ptr(), value as u8, idx.0) };
    }
    for (i, op) in program.ops.iter().enumerate() {
        let out = Idx(i as u32);
        // Inputs already materialized above; re-running their `Create` is
        // harmless (idempotent), but skip it so a caller-supplied input
        // value is never silently overwritten by the recorded constant.
        if program.inputs.contains(&out) {
            continue;
        }
        unsafe {
            match *op {
                Op::Create(val) => cirrus_rt_create(buf.as_mut_ptr(), val as u8, out.0),
                Op::BitAnd(a, b) => cirrus_rt_bitand(buf.as_mut_ptr(), a.0, b.0, out.0),
                Op::BitOr(a, b) => cirrus_rt_bitor(buf.as_mut_ptr(), a.0, b.0, out.0),
                Op::BitXor(a, b) => cirrus_rt_bitxor(buf.as_mut_ptr(), a.0, b.0, out.0),
                Op::Mux { cond, then, r#else } => {
                    cirrus_rt_mux(buf.as_mut_ptr(), cond.0, then.0, r#else.0, out.0)
                }
            }
        }
    }
    program
        .outputs
        .iter()
        .map(|idx| buf[idx.get()] != 0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cirrus_core::{ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate};
    use cirrus_recompile_core::Recorder;

    #[test]
    fn execute_matches_the_recorder_s_own_interpreter() {
        let mut recorder = Recorder::new();
        let a = recorder.create(true).unwrap();
        let b = recorder.create(false).unwrap();
        let and = ContextWithBitAnd::bitand(&mut recorder, a, b).unwrap();
        let or = ContextWithBitOr::bitor(&mut recorder, a, b).unwrap();
        let xor = ContextWithBitXor::bitxor(&mut recorder, a, b).unwrap();
        let program = recorder.finish(vec![a, b], vec![and, or, xor]);

        for &(x, y) in &[(false, false), (false, true), (true, false), (true, true)] {
            let expected = cirrus_recompile_core::interpret(&program, &[x, y]);
            let actual = execute(&program, &[x, y]);
            assert_eq!(actual, expected);
        }
    }
}
