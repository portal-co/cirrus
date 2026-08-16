//! Deoptimize secret-dependent early-exit loops back into their pre-
//! short-circuit, always-runs-every-iteration form.
//!
//! `cirrus-llvm-frontend`'s interpreter hard-errors the moment it meets a
//! conditional branch whose condition it cannot resolve concretely (see
//! `cirrus-llvm-frontend::execute_function`'s `branch`/`switch` handling).
//! That is correct in general, but it means a very common guest idiom --
//! comparing two buffers for equality, or bulk-reducing over an array, with
//! a data-dependent `break` once the answer is known -- fails to lower at
//! all, even though the loop's own trip count is concrete/public. Before an
//! optimizing compiler's short-circuit transform, that same code ran every
//! iteration unconditionally, folding the result via a boolean select
//! instead of branching.
//!
//! This pass recognizes a narrow, structurally provable instance of that
//! idiom and rewrites it back to the always-runs-every-iteration form:
//!
//! ```text
//! loop.latch:                       loop.latch:
//!   %i.next = ...                     %i.next = ...
//!   %guard = icmp ... %i.next, %n     %guard = icmp ... %i.next, %n
//!   br %guard, %H, %natural_exit      br %guard, %H, %natural_exit
//! H:                                 H:
//!   ...                                %acc = phi [%init, preheader],
//!   %cond = <data-dependent>                     [%acc.next, loop.latch]
//!   br %cond, %exit_succ, %cont       ...
//! %exit_succ:                          %cond = <data-dependent>
//!   br %merge                          %acc.next = select %cond, %exitval, %acc
//! %cont:                               br %cont          ; unconditional now
//! ...                                %exit_succ:           ; now dead code
//! %merge:                              br %merge
//! %result = phi [%exitval, %exit_succ],  %cont:
//!            [%naturalval, loop.latch] ...
//!                                    %merge:
//!                                      %result = phi [%exitval, %exit_succ],
//!                                                 [%acc, loop.latch]
//! ```
//!
//! Anything outside this precise shape is left completely untouched --
//! this pass is purely additive, never a silent miscompile risk. Everything
//! it declines to transform still hits `cirrus-llvm-frontend`'s existing
//! hard error, unchanged.
//!
//! # Idiom contract
//!
//! 1. A natural loop with a single latch `L`, whose own terminator is a
//!    concrete-bounded conditional branch: one successor is the loop header
//!    `H` (the back edge), the other (`natural_exit`) leaves the loop
//!    directly -- no chasing on that side.
//! 2. Exactly one *other* conditional branch anywhere in the loop body
//!    (`branch_block`), found by scanning the whole body up front. Zero or
//!    more than one candidate anywhere in the body -- including any other
//!    conditional branch that isn't itself an in/out split, or a nested
//!    loop -- rejects the whole loop untouched.
//! 3. That branch's two successors split exactly one-in/one-out of the
//!    loop body: `continue_target` stays in the body, `exit_target` leaves
//!    it. Chasing `exit_target` through only empty (terminator-only)
//!    blocks, up to a small bound, must land exactly on `natural_exit`.
//! 4. `natural_exit` has exactly one phi, with exactly two incoming edges:
//!    one from `L` (`natural_value`) and one from the exit chain's last
//!    block (`exit_value`). Both must be constants or otherwise defined
//!    outside the loop body -- never a value computed this iteration.

use std::collections::{HashMap, HashSet};

use inkwell::basic_block::BasicBlock;
use inkwell::values::{
    BasicValue, BasicValueEnum, FunctionValue, InstructionOpcode, InstructionValue, Operand,
    PhiValue,
};

/// Bound on how many empty (terminator-only) blocks the early-exit chain
/// may pass through before it must land on the loop's own natural exit.
const MAX_EXIT_CHAIN: u32 = 4;

/// Attempt the recognizer/rewrite on every natural loop in `function`.
/// Returns `true` if any loop was rewritten.
pub fn deloopify_early_exits<'ctx>(function: FunctionValue<'ctx>) -> bool {
    let Some(entry) = function.get_first_basic_block() else {
        return false;
    };

    let mut changed = false;
    // Candidates are recomputed after each rewrite: a rewrite changes the
    // CFG (redirects branch_block's terminator), and this pass never needs
    // to revisit an already-rewritten region, so a fixed-point loop over a
    // shrinking candidate set terminates quickly for the tiny functions
    // this pass targets.
    loop {
        let (successors, predecessors) = build_cfg_maps(function);
        let back_edges = find_back_edges(entry, &successors);
        let mut by_header: HashMap<BasicBlock<'ctx>, Vec<BasicBlock<'ctx>>> = HashMap::new();
        for (latch, header) in &back_edges {
            by_header.entry(*header).or_default().push(*latch);
        }

        let mut rewrote_one = false;
        for (header, latches) in &by_header {
            if latches.len() != 1 {
                continue; // multi-latch/irreducible: not handled.
            }
            let latch = latches[0];
            let body = natural_loop_body(*header, latch, &predecessors);
            // Reject nested loops: another back edge whose header sits
            // strictly inside this loop's body.
            if back_edges
                .iter()
                .any(|(other_latch, other_header)| {
                    *other_header != *header && *other_latch != latch && body.contains(other_header)
                })
            {
                continue;
            }
            if let Some(site) = classify(*header, latch, &body, &successors) {
                rewrite(&site);
                changed = true;
                rewrote_one = true;
                break; // CFG changed; recompute before looking further.
            }
        }
        if !rewrote_one {
            break;
        }
    }
    changed
}

fn terminator_successors<'ctx>(block: BasicBlock<'ctx>) -> Vec<BasicBlock<'ctx>> {
    let Some(terminator) = block.get_terminator() else {
        return Vec::new();
    };
    match terminator.get_num_operands() {
        1 => terminator
            .get_operand(0)
            .and_then(Operand::block)
            .into_iter()
            .collect(),
        3 => [terminator.get_operand(1), terminator.get_operand(2)]
            .into_iter()
            .filter_map(|operand| operand.and_then(Operand::block))
            .collect(),
        _ => Vec::new(), // switch/indirectbr/ret/etc. -- not chased.
    }
}

fn build_cfg_maps<'ctx>(
    function: FunctionValue<'ctx>,
) -> (
    HashMap<BasicBlock<'ctx>, Vec<BasicBlock<'ctx>>>,
    HashMap<BasicBlock<'ctx>, Vec<BasicBlock<'ctx>>>,
) {
    let mut successors = HashMap::new();
    let mut predecessors: HashMap<BasicBlock<'ctx>, Vec<BasicBlock<'ctx>>> = HashMap::new();
    for block in function.get_basic_blocks() {
        let s = terminator_successors(block);
        for &target in &s {
            predecessors.entry(target).or_default().push(block);
        }
        successors.insert(block, s);
    }
    (successors, predecessors)
}

/// DFS back-edge detection. Returns `(latch, header)` pairs.
fn find_back_edges<'ctx>(
    entry: BasicBlock<'ctx>,
    successors: &HashMap<BasicBlock<'ctx>, Vec<BasicBlock<'ctx>>>,
) -> Vec<(BasicBlock<'ctx>, BasicBlock<'ctx>)> {
    let mut visited = HashSet::new();
    let mut on_stack = HashSet::new();
    let mut back_edges = Vec::new();
    let empty = Vec::new();
    let mut stack: Vec<(BasicBlock<'ctx>, std::slice::Iter<'_, BasicBlock<'ctx>>)> = Vec::new();

    visited.insert(entry);
    on_stack.insert(entry);
    stack.push((entry, successors.get(&entry).unwrap_or(&empty).iter()));

    while let Some((node, iter)) = stack.last_mut() {
        let node = *node;
        if let Some(&successor) = iter.next() {
            if on_stack.contains(&successor) {
                back_edges.push((node, successor));
            } else if visited.insert(successor) {
                on_stack.insert(successor);
                stack.push((successor, successors.get(&successor).unwrap_or(&empty).iter()));
            }
        } else {
            on_stack.remove(&node);
            stack.pop();
        }
    }
    back_edges
}

/// The standard natural-loop node set for back edge `latch -> header`:
/// `header` plus every block that can reach `latch` without passing back
/// through `header`.
fn natural_loop_body<'ctx>(
    header: BasicBlock<'ctx>,
    latch: BasicBlock<'ctx>,
    predecessors: &HashMap<BasicBlock<'ctx>, Vec<BasicBlock<'ctx>>>,
) -> HashSet<BasicBlock<'ctx>> {
    let mut body = HashSet::new();
    body.insert(header);
    body.insert(latch);
    let mut stack = vec![latch];
    while let Some(block) = stack.pop() {
        if block == header {
            continue;
        }
        for &predecessor in predecessors.get(&block).into_iter().flatten() {
            if body.insert(predecessor) {
                stack.push(predecessor);
            }
        }
    }
    body
}

struct RecognizedSite<'ctx> {
    header: BasicBlock<'ctx>,
    latch: BasicBlock<'ctx>,
    branch: InstructionValue<'ctx>,
    condition: BasicValueEnum<'ctx>,
    /// `true` when the branch condition being *true* takes the exit.
    exit_when_true: bool,
    continue_target: BasicBlock<'ctx>,
    exit_value_block: BasicBlock<'ctx>,
    exit_value: BasicValueEnum<'ctx>,
    natural_value: BasicValueEnum<'ctx>,
    merge: BasicBlock<'ctx>,
    merge_phi: PhiValue<'ctx>,
}

fn conditional_branch_parts<'ctx>(
    terminator: InstructionValue<'ctx>,
) -> Option<(BasicValueEnum<'ctx>, BasicBlock<'ctx>, BasicBlock<'ctx>)> {
    if terminator.get_num_operands() != 3 {
        return None;
    }
    let condition = terminator.get_operand(0).and_then(Operand::value)?;
    // LLVM's low-level operand list stores the false successor before the
    // true successor (matching cirrus-llvm-frontend's own `branch()`).
    let false_target = terminator.get_operand(1).and_then(Operand::block)?;
    let true_target = terminator.get_operand(2).and_then(Operand::block)?;
    Some((condition, false_target, true_target))
}

fn is_empty_block(block: BasicBlock<'_>) -> bool {
    match (block.get_first_instruction(), block.get_terminator()) {
        (Some(first), Some(terminator)) => first == terminator,
        _ => false,
    }
}

/// Chase an unconditional-branch-only, empty-block chain from `start`,
/// requiring it to land exactly on `merge` within `MAX_EXIT_CHAIN` hops.
/// Returns the final block (`merge`'s real predecessor along this chain).
fn chase_empty_chain<'ctx>(
    predecessor: BasicBlock<'ctx>,
    mut current: BasicBlock<'ctx>,
    merge: BasicBlock<'ctx>,
) -> Option<BasicBlock<'ctx>> {
    let mut previous = predecessor;
    for _ in 0..MAX_EXIT_CHAIN {
        if current == merge {
            return Some(previous);
        }
        if !is_empty_block(current) {
            return None;
        }
        let terminator = current.get_terminator()?;
        if terminator.get_num_operands() != 1 {
            return None;
        }
        let next = terminator.get_operand(0).and_then(Operand::block)?;
        previous = current;
        current = next;
    }
    (current == merge).then_some(previous)
}

fn defined_outside<'ctx>(value: BasicValueEnum<'ctx>, body: &HashSet<BasicBlock<'ctx>>) -> bool {
    match value.as_instruction_value() {
        Some(instruction) => match instruction.get_parent() {
            Some(parent) => !body.contains(&parent),
            None => true,
        },
        None => true, // constant, global, or function argument.
    }
}

fn classify<'ctx>(
    header: BasicBlock<'ctx>,
    latch: BasicBlock<'ctx>,
    body: &HashSet<BasicBlock<'ctx>>,
    successors: &HashMap<BasicBlock<'ctx>, Vec<BasicBlock<'ctx>>>,
) -> Option<RecognizedSite<'ctx>> {
    let latch_terminator = latch.get_terminator()?;
    let (_, false_target, true_target) = conditional_branch_parts(latch_terminator)?;
    let natural_exit = if false_target == header {
        true_target
    } else if true_target == header {
        false_target
    } else {
        return None;
    };
    if body.contains(&natural_exit) {
        return None;
    }

    let mut candidate: Option<(BasicBlock<'ctx>, InstructionValue<'ctx>)> = None;
    for &block in body {
        if block == latch {
            continue;
        }
        let Some(terminator) = block.get_terminator() else {
            continue;
        };
        if terminator.get_num_operands() == 1 {
            continue; // plain unconditional jump: fine, not a candidate.
        }
        if terminator.get_num_operands() != 3 {
            return None; // switch/indirectbr in the body: unsupported.
        }
        let block_successors = successors.get(&block).cloned().unwrap_or_default();
        let (in_body, out_body): (Vec<_>, Vec<_>) =
            block_successors.into_iter().partition(|s| body.contains(s));
        if in_body.len() == 1 && out_body.len() == 1 {
            if candidate.is_some() {
                return None; // a second real candidate: reject the loop.
            }
            candidate = Some((block, terminator));
        } else {
            return None; // both-in or both-out: unsupported extra branching.
        }
    }
    let (branch_block, branch) = candidate?;
    let (condition, false_target, true_target) = conditional_branch_parts(branch)?;
    let (continue_target, exit_target, exit_when_true) = if body.contains(&false_target) {
        (false_target, true_target, true)
    } else {
        (true_target, false_target, false)
    };
    if !body.contains(&continue_target) || body.contains(&exit_target) {
        return None;
    }

    let exit_value_block = chase_empty_chain(branch_block, exit_target, natural_exit)?;

    let merge = natural_exit;
    let first = merge.get_first_instruction()?;
    if first.get_opcode() != InstructionOpcode::Phi {
        return None;
    }
    let merge_phi = PhiValue::try_from(first).ok()?;
    if merge_phi.count_incoming() != 2 {
        return None;
    }
    let mut exit_value = None;
    let mut natural_value = None;
    for index in 0..merge_phi.count_incoming() {
        let (value, incoming_block) = merge_phi.get_incoming(index)?;
        if incoming_block == exit_value_block {
            exit_value = Some(value);
        } else if incoming_block == latch {
            natural_value = Some(value);
        } else {
            return None;
        }
    }
    let exit_value = exit_value?;
    let natural_value = natural_value?;
    if !defined_outside(exit_value, body) || !defined_outside(natural_value, body) {
        return None;
    }

    Some(RecognizedSite {
        header,
        latch,
        branch,
        condition,
        exit_when_true,
        continue_target,
        exit_value_block,
        exit_value,
        natural_value,
        merge,
        merge_phi,
    })
}

fn first_non_phi_position(block: BasicBlock<'_>) -> InstructionValue<'_> {
    for instruction in block.get_instructions() {
        if instruction.get_opcode() != InstructionOpcode::Phi {
            return instruction;
        }
    }
    block.get_terminator().expect("every block has a terminator")
}

fn rewrite(site: &RecognizedSite<'_>) {
    let context = site.header.get_context();
    let builder = context.create_builder();

    let phi_type = site.merge_phi.as_basic_value().get_type();
    builder.position_before(&first_non_phi_position(site.header));
    let Ok(acc_phi) = builder.build_phi(phi_type, "deloopify.acc") else {
        return;
    };
    for predecessor in predecessors_excluding(site.header, site.latch) {
        acc_phi.add_incoming(&[(&site.natural_value, predecessor)]);
    }

    builder.position_before(&site.branch);
    let condition = site.condition.into_int_value();
    let (then_value, else_value) = if site.exit_when_true {
        (site.exit_value, acc_phi.as_basic_value())
    } else {
        (acc_phi.as_basic_value(), site.exit_value)
    };
    let Ok(select_result) = builder.build_select(condition, then_value, else_value, "deloopify.select")
    else {
        return;
    };
    acc_phi.add_incoming(&[(&select_result, site.latch)]);

    let Ok(_) = builder.build_unconditional_branch(site.continue_target) else {
        return;
    };
    site.branch.erase_from_basic_block();

    builder.position_before(&first_non_phi_position(site.merge));
    let Ok(new_phi) = builder.build_phi(phi_type, "deloopify.result") else {
        return;
    };
    // The merge's back-edge incoming must be `select_result` (this
    // iteration's freshly updated value), not `acc_phi` (the value from
    // *before* this iteration's update) -- they only coincide when the
    // final iteration didn't itself trigger the exit.
    new_phi.add_incoming(&[
        (&site.exit_value, site.exit_value_block),
        (&select_result, site.latch),
    ]);
    site.merge_phi.replace_all_uses_with(&new_phi);
    site.merge_phi.as_instruction().erase_from_basic_block();
}

fn predecessors_excluding<'ctx>(
    block: BasicBlock<'ctx>,
    excluded: BasicBlock<'ctx>,
) -> Vec<BasicBlock<'ctx>> {
    let Some(function) = block.get_parent() else {
        return Vec::new();
    };
    function
        .get_basic_blocks()
        .into_iter()
        .filter(|candidate| *candidate != excluded && terminator_successors(*candidate).contains(&block))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cirrus_llvm_frontend::{
        ArgumentBinding, Export, HostCallRegistry, LowerRequest, LoweringLimits, RecorderBackend,
        RegionBinding, RegionByte, ScalarBinding, lower_module,
    };
    use cirrus_recompile_core::interpret;
    use inkwell::context::Context;
    use inkwell::memory_buffer::MemoryBuffer;

    /// A hand-written equivalent of `-O2`'s early-exit lowering of
    /// `for (i = 0; i < n; i++) if (a[i] != b[i]) return 0; return 1;`.
    const MEMCMP_KERNEL: &str = "\
        define i1 @kernel(ptr %a, ptr %b, i32 %n) {\n\
        entry:\n\
          br label %header\n\
        header:\n\
          %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]\n\
          br label %body\n\
        body:\n\
          %pa = getelementptr i8, ptr %a, i32 %i\n\
          %pb = getelementptr i8, ptr %b, i32 %i\n\
          %va = load i8, ptr %pa\n\
          %vb = load i8, ptr %pb\n\
          %eq = icmp eq i8 %va, %vb\n\
          br i1 %eq, label %continue, label %exit.trampoline\n\
        exit.trampoline:\n\
          br label %merge\n\
        continue:\n\
          %i.next = add i32 %i, 1\n\
          br label %latch\n\
        latch:\n\
          %cont = icmp ult i32 %i.next, %n\n\
          br i1 %cont, label %header, label %merge\n\
        merge:\n\
          %result = phi i1 [ false, %exit.trampoline ], [ true, %latch ]\n\
          ret i1 %result\n\
        }\n\
        ";

    fn parse<'ctx>(context: &'ctx Context, source: &str) -> inkwell::module::Module<'ctx> {
        context
            .create_module_from_ir(MemoryBuffer::create_from_memory_range_copy(
                source.as_bytes(),
                "deloopify-test.ll",
            ))
            .expect("valid test IR")
    }

    fn request(len: usize) -> LowerRequest<'static, RecorderBackend> {
        let arguments: &'static [ArgumentBinding] = Box::leak(Box::new([
            ArgumentBinding::Region(RegionBinding {
                bytes: vec![RegionByte::Symbolic; len],
                writable: false,
            }),
            ArgumentBinding::Region(RegionBinding {
                bytes: vec![RegionByte::Symbolic; len],
                writable: false,
            }),
            ArgumentBinding::Scalar(ScalarBinding::Concrete(len as u64)),
        ]));
        let exports: &'static [Export] = Box::leak(Box::new([Export::Return]));
        let hosts = Box::leak(Box::new(HostCallRegistry::new()));
        LowerRequest {
            entry: "kernel",
            arguments,
            globals: &[],
            exports,
            limits: LoweringLimits::default(),
            host_calls: hosts,
        }
    }

    /// Low-bit-first bits for `a` then `b`, matching how two fully-symbolic
    /// `RegionByte::Symbolic` regions consume `Program::inputs`.
    fn input_bits(a: &[u8], b: &[u8]) -> Vec<bool> {
        a.iter()
            .chain(b.iter())
            .flat_map(|&byte| (0..8).map(move |bit| (byte >> bit) & 1 != 0))
            .collect()
    }

    fn run_deloopified(a: &[u8], b: &[u8]) -> bool {
        assert_eq!(a.len(), b.len());
        let context = Context::create();
        let module = parse(&context, MEMCMP_KERNEL);
        let function = module.get_function("kernel").expect("kernel exists");
        assert!(deloopify_early_exits(function), "expected a rewrite");

        let program = lower_module(&module, &request(a.len())).expect("lowers after deloopify");
        let output = interpret(&program, &input_bits(a, b));
        assert_eq!(output.len(), 1);
        output[0]
    }

    #[test]
    fn undeoptimized_kernel_rejects_the_symbolic_compare() {
        let context = Context::create();
        let module = parse(&context, MEMCMP_KERNEL);
        let error = lower_module(&module, &request(4)).unwrap_err();
        assert!(error.to_string().contains("symbolic branch"));
    }

    #[test]
    fn deoptimized_kernel_recognizes_equal_buffers_across_every_length() {
        for len in 1..=8usize {
            let data = std::vec![7u8; len];
            assert!(run_deloopified(&data, &data), "len={len}");
        }
    }

    #[test]
    fn deoptimized_kernel_recognizes_a_mismatch_at_every_position() {
        let len = 6usize;
        for mismatch_at in 0..len {
            let a: Vec<u8> = (0..len as u8).collect();
            let mut b = a.clone();
            b[mismatch_at] = b[mismatch_at].wrapping_add(1);
            assert!(!run_deloopified(&a, &b), "mismatch_at={mismatch_at}");
        }
    }

    /// Two independent secret-dependent branches in one loop body: the
    /// idiom contract requires exactly one candidate, so the pass must
    /// leave this loop untouched and lowering must still fail exactly as
    /// it does today.
    const TWO_CANDIDATE_KERNEL: &str = "\
        define i1 @kernel(ptr %a, ptr %b, i32 %n) {\n\
        entry:\n\
          br label %header\n\
        header:\n\
          %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]\n\
          br label %body\n\
        body:\n\
          %pa = getelementptr i8, ptr %a, i32 %i\n\
          %pb = getelementptr i8, ptr %b, i32 %i\n\
          %va = load i8, ptr %pa\n\
          %vb = load i8, ptr %pb\n\
          %eq = icmp eq i8 %va, %vb\n\
          br i1 %eq, label %second, label %exit.trampoline\n\
        second:\n\
          %ne = icmp ne i8 %va, %vb\n\
          br i1 %ne, label %exit.trampoline2, label %continue\n\
        exit.trampoline:\n\
          br label %merge\n\
        exit.trampoline2:\n\
          br label %merge\n\
        continue:\n\
          %i.next = add i32 %i, 1\n\
          br label %latch\n\
        latch:\n\
          %cont = icmp ult i32 %i.next, %n\n\
          br i1 %cont, label %header, label %merge\n\
        merge:\n\
          %result = phi i1 [ false, %exit.trampoline ], [ false, %exit.trampoline2 ], [ true, %latch ]\n\
          ret i1 %result\n\
        }\n\
        ";

    #[test]
    fn a_second_branch_in_the_body_is_left_untouched() {
        let context = Context::create();
        let module = parse(&context, TWO_CANDIDATE_KERNEL);
        let function = module.get_function("kernel").expect("kernel exists");
        assert!(!deloopify_early_exits(function), "must not rewrite");
        let error = lower_module(&module, &request(2)).unwrap_err();
        assert!(error.to_string().contains("symbolic branch"));
    }
}
