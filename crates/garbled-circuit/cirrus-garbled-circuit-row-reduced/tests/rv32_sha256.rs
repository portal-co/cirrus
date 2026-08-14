use core::{array, convert::Infallible};
use std::{
    env, fs,
    path::PathBuf,
    process::{Command, Output},
    string::String,
    time::Instant,
    vec::Vec,
};

use cirrus_core::Pusher;
use cirrus_ert::{DefaultHandler, RawMemory, RvDefaultHandler, ert_func};
use cirrus_ert_sha256_fixture::sha256_compress;
use cirrus_garbled_circuit_row_reduced::{Evaluator, GC, GarblingRecord, Label};
use digest::{OutputSizeUser, array::Array};
use sha2::Sha256;

const BASE: u32 = 0x8000_0000;
const INPUT: [u32; 16] = [0x6162_6380, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24];
// The RV32IM workload touches 1,248 labels; 4,096 leaves a stable margin
// while keeping the host replay representative of an embedded symbolic-stack
// budget.
const STACK_SLOTS: usize = 4_096;

struct Records<const N: usize>(Vec<[[u8; N]; 3]>);

impl<const N: usize> Records<N> {
    fn new() -> Self {
        Self(Vec::new())
    }
}

impl<const N: usize> Pusher<[[u8; N]; 3]> for Records<N> {
    fn push(&mut self, table: [[u8; N]; 3]) {
        self.0.push(table);
    }
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .and_then(|path| path.parent())
        .expect("the row-reduced crate lives below the workspace root")
        .to_path_buf()
}

fn assert_success(action: &str, output: &Output) {
    assert!(
        output.status.success(),
        "{action} failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn ensure_target(target: &str) {
    let installed = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .expect("rustup is required to inspect installed targets");
    assert_success("listing Rust targets", &installed);
    if installed
        .stdout
        .split(|byte| *byte == b'\n')
        .any(|installed| installed == target.as_bytes())
    {
        return;
    }
    let install = Command::new("rustup")
        .args(["target", "add", target])
        .output()
        .expect("rustup is required to install the self-test target");
    assert_success("installing the bare-metal target", &install);
}

fn build_self_test() -> PathBuf {
    const PACKAGE: &str = "cirrus-ert-selftest";
    const TARGET: &str = "riscv32im-unknown-none-elf";
    ensure_target(TARGET);
    let root = workspace_root();
    let target_dir = root
        .join("target")
        .join("cirrus-row-reduced-self-test")
        .join(PACKAGE);
    let build = Command::new("cargo")
        .current_dir(&root)
        .args([
            "build",
            "-p",
            PACKAGE,
            "--features",
            "bare-metal",
            "--target",
            TARGET,
            "--release",
            "--target-dir",
        ])
        .arg(&target_dir)
        .env("RUSTFLAGS", "-C panic=abort")
        .output()
        .expect("cargo must build the bare-metal self-test");
    assert_success("building the bare-metal self-test", &build);
    target_dir.join(TARGET).join("release").join(PACKAGE)
}

fn elf_u16(image: &[u8], offset: usize) -> usize {
    u16::from_le_bytes(
        image[offset..offset + 2]
            .try_into()
            .expect("ELF header is complete"),
    ) as usize
}

fn elf_u32(image: &[u8], offset: usize) -> usize {
    u32::from_le_bytes(
        image[offset..offset + 4]
            .try_into()
            .expect("ELF header is complete"),
    ) as usize
}

fn elf_header(image: &[u8]) -> (usize, usize, usize, usize) {
    assert_eq!(&image[..4], b"\x7fELF", "image is an ELF file");
    assert_eq!(image[4], 1, "self-test ELF is 32-bit");
    assert_eq!(image[5], 1, "self-test ELF is little-endian");
    (
        elf_u32(image, 32),
        elf_u16(image, 46),
        elf_u16(image, 48),
        elf_u16(image, 50),
    )
}

fn section_header(image: &[u8], index: usize) -> usize {
    let (offset, size, count, _) = elf_header(image);
    assert!(index < count, "section index is in range");
    offset + index * size
}

fn section_name<'a>(image: &'a [u8], header: usize) -> &'a str {
    let (_, _, _, strings) = elf_header(image);
    let strings = section_header(image, strings);
    let strings_offset = elf_u32(image, strings + 16);
    let strings_size = elf_u32(image, strings + 20);
    let start = strings_offset + elf_u32(image, header);
    let bytes = &image[start..strings_offset + strings_size];
    let length = bytes
        .iter()
        .position(|byte| *byte == 0)
        .expect("ELF section name is terminated");
    core::str::from_utf8(&bytes[..length]).expect("ELF section name is UTF-8")
}

fn selected_section(image: &[u8], wanted: &str) -> (usize, usize, usize) {
    let (_, _, count, _) = elf_header(image);
    for index in 0..count {
        let header = section_header(image, index);
        if section_name(image, header) == wanted {
            return (
                elf_u32(image, header + 12),
                elf_u32(image, header + 16),
                elf_u32(image, header + 20),
            );
        }
    }
    panic!("ELF image lacks required section {wanted}");
}

fn mapped_image(image: &[u8], names: &[&str]) -> Vec<u8> {
    let selected: Vec<_> = names
        .iter()
        .map(|name| selected_section(image, name))
        .collect();
    let end = selected
        .iter()
        .map(|(address, _, size)| address + size)
        .max()
        .expect("at least one ELF section is mapped");
    let base = BASE as usize;
    assert!(selected.iter().all(|(address, _, _)| *address >= base));
    let mut mapping = vec![0; end - base];
    for (address, offset, size) in selected {
        mapping[address - base..address - base + size]
            .copy_from_slice(&image[offset..offset + size]);
    }
    mapping
}

fn symbol_address(image: &[u8], wanted: &str) -> u32 {
    const SYMBOL_TABLE: usize = 2;
    let (_, _, count, _) = elf_header(image);
    for index in 0..count {
        let header = section_header(image, index);
        if elf_u32(image, header + 4) != SYMBOL_TABLE {
            continue;
        }
        let strings = section_header(image, elf_u32(image, header + 24));
        let strings_offset = elf_u32(image, strings + 16);
        let strings_size = elf_u32(image, strings + 20);
        let symbols_offset = elf_u32(image, header + 16);
        let symbols_size = elf_u32(image, header + 20);
        let entry_size = elf_u32(image, header + 36);
        assert_eq!(entry_size, 16, "self-test uses ELF32 symbol entries");
        for entry in (symbols_offset..symbols_offset + symbols_size).step_by(entry_size) {
            let name_offset = strings_offset + elf_u32(image, entry);
            let name_bytes = &image[name_offset..strings_offset + strings_size];
            let length = name_bytes
                .iter()
                .position(|byte| *byte == 0)
                .expect("symbol name is terminated");
            if core::str::from_utf8(&name_bytes[..length]).expect("symbol name is UTF-8") == wanted
            {
                return elf_u32(image, entry + 4) as u32;
            }
        }
    }
    panic!("ELF image lacks required symbol {wanted}");
}

fn mapped_memory(mapping: &[u8]) -> RawMemory<'static> {
    // SAFETY: `BASE` maps the contiguous selected guest sections onto
    // `mapping`; the self-test executes only selected code and static data.
    unsafe { RawMemory::new(mapping.as_ptr().wrapping_sub(BASE as usize), None) }
}

fn input_zero_label(word: usize, bit: usize) -> [u8; 16] {
    let wire = u16::try_from(word * 32 + bit + 1).expect("the test has few enough input wires");
    let mut label = [0; 16];
    label[..2].copy_from_slice(&wire.to_le_bytes());
    label
}

fn word(value: u32, word: usize, one: [u8; 16]) -> [[u8; 16]; 32] {
    array::from_fn(|bit| {
        let zero = input_zero_label(word, bit);
        if value & (1 << bit) == 0 {
            zero
        } else {
            array::from_fn(|index| zero[index] ^ one[index])
        }
    })
}

fn garbling_args() -> [([Label<16>; 32], Option<u32>); 16] {
    array::from_fn(|word| {
        (
            array::from_fn(|bit| Label::new(input_zero_label(word, bit))),
            None,
        )
    })
}

fn evaluation_args(one: [u8; 16]) -> [([[u8; 16]; 32], Option<u32>); 16] {
    array::from_fn(|word_index| (word(INPUT[word_index], word_index, one), None))
}

fn no_hash<C>(_: &mut C, _: &[[Label<16>; 32]]) -> Result<[u8; 32], Infallible> {
    Ok([0; 32])
}

fn evaluator_no_hash<C>(
    _: &mut C,
    _: &[[[u8; 16]; 32]],
) -> Result<[u8; 32], cirrus_garbled_circuit_row_reduced::EvaluationError> {
    Ok([0; 32])
}

fn bool_word(value: u32) -> [bool; 32] {
    array::from_fn(|bit| value & (1 << bit) != 0)
}

fn bool_no_hash<C>(_: &mut C, _: &[[bool; 32]]) -> Result<[u8; 32], Infallible> {
    Ok([0; 32])
}

#[test]
fn rv32_sha256_replays_the_three_row_stream_and_reports_traffic() {
    let image = build_self_test();
    let elf = fs::read(image).expect("built RV32 ELF is readable");
    let mapping = mapped_image(&elf, &[".text.ert_workload", ".rodata.ert_workload"]);
    let entry = symbol_address(&elf, "__ert_workload_entry");
    let memory = mapped_memory(&mapping);
    let expected = sha256_compress(
        INPUT[0], INPUT[1], INPUT[2], INPUT[3], INPUT[4], INPUT[5], INPUT[6], INPUT[7], INPUT[8],
        INPUT[9], INPUT[10], INPUT[11], INPUT[12], INPUT[13], INPUT[14], INPUT[15],
    );
    let zero = [0; 16];
    let one = array::from_fn(|index| (index == 0) as u8);
    let delta: Array<u8, <Sha256 as OutputSizeUser>::OutputSize> =
        array::from_fn(|index| (index == 0) as u8).into();
    let garbling_zero = Label::new(zero);
    let garbling_one = garbling_zero.not();

    let mut bool_registers = [[false; 32]; 32];
    let mut bool_constants = [None; 32];
    let mut bool_rstack = [0; 512];
    let mut bool_vstack = vec![false; STACK_SLOTS];
    let mut bool_handler = RvDefaultHandler {
        inner: DefaultHandler {
            context: (),
            hash: bool_no_hash,
        },
    };
    let bool_result = ert_func::<_, _, 16, 2>(
        &mut bool_handler,
        memory,
        &mut bool_rstack,
        &mut bool_vstack,
        entry,
        &mut bool_registers,
        &mut bool_constants,
        false,
        true,
        INPUT.map(|value| (bool_word(value), None)),
    );
    assert!(
        bool_result.is_ok(),
        "the host-mapped RV32 image executes natively"
    );
    let bool_result = match bool_result {
        Ok(result) => result,
        Err(_) => unreachable!("the result was checked above"),
    };
    assert_eq!(bool_result[1], (bool_word(expected), None));

    let mut garbled_registers = [[garbling_zero; 32]; 32];
    let mut garbled_constants = [None; 32];
    let mut garbled_rstack = [0; 512];
    let stack_sentinel = Label::new([0xa5; 16]);
    let mut garbled_vstack = vec![stack_sentinel; STACK_SLOTS];
    let mut records = Records::new();
    let garbler = GC::<Sha256, 16>::new(&mut records, delta);
    let mut handler = RvDefaultHandler {
        inner: DefaultHandler {
            context: garbler,
            hash: no_hash,
        },
    };

    let started = Instant::now();
    let garbled = ert_func::<_, _, 16, 2>(
        &mut handler,
        memory,
        &mut garbled_rstack,
        &mut garbled_vstack,
        entry,
        &mut garbled_registers,
        &mut garbled_constants,
        garbling_zero,
        garbling_one,
        garbling_args(),
    );
    let garbling_elapsed = started.elapsed();
    assert!(
        garbled.is_ok(),
        "the locked RV32 workload garbles successfully"
    );
    let garbled = match garbled {
        Ok(result) => result,
        Err(_) => unreachable!("the result was checked above"),
    };
    drop(handler);
    let tables = records.0.len();
    assert_eq!(tables, 358_656, "the locked workload's AND count is stable");
    let touched_stack_slots = garbled_vstack
        .iter()
        .position(|label| *label != stack_sentinel)
        .map(|lowest_touched| STACK_SLOTS - lowest_touched)
        .unwrap_or(0);
    let return_stack_slots = garbled_rstack
        .iter()
        .rposition(|slot| *slot != 0)
        .map(|last_touched| last_touched + 1)
        .unwrap_or(0);
    assert_eq!(core::mem::size_of::<Label<16>>(), 16);
    assert_eq!(touched_stack_slots, 1_248);
    assert_eq!(return_stack_slots, 2);

    let mut evaluated_registers = [[zero; 32]; 32];
    let mut evaluated_constants = [None; 32];
    let mut evaluated_rstack = [0; 512];
    let mut evaluated_vstack = vec![[0xa5; 16]; STACK_SLOTS];
    let evaluator =
        Evaluator::<Sha256, _, 16>::new(records.0.into_iter().map(GarblingRecord::Table));
    let mut handler = RvDefaultHandler {
        inner: DefaultHandler {
            context: evaluator,
            hash: evaluator_no_hash,
        },
    };
    let started = Instant::now();
    let evaluated = ert_func::<_, _, 16, 2>(
        &mut handler,
        memory,
        &mut evaluated_rstack,
        &mut evaluated_vstack,
        entry,
        &mut evaluated_registers,
        &mut evaluated_constants,
        zero,
        one,
        evaluation_args(one),
    );
    let evaluation_elapsed = started.elapsed();
    assert!(
        evaluated.is_ok(),
        "the evaluator replays every ordered record"
    );
    let evaluated = match evaluated {
        Ok(result) => result,
        Err(_) => unreachable!("the result was checked above"),
    };

    assert_eq!(garbled[0].1, Some(u32::MAX));
    assert_eq!(evaluated[0].1, Some(u32::MAX));
    assert_eq!(garbled[1].1, None);
    assert_eq!(evaluated[1].1, None);
    for bit in 0..32 {
        assert_eq!(
            evaluated[1].0[bit],
            if expected & (1 << bit) == 0 {
                garbled[1].0[bit].zero_label()
            } else {
                array::from_fn(|index| {
                    garbled[1].0[bit].zero_label()[index] ^ if index == 0 { 1 } else { 0 }
                })
            },
            "SHA-256 result bit {bit}",
        );
    }

    let table_bytes = tables * 3 * 16;
    assert_eq!(table_bytes, 17_215_488);
    eprintln!(
        "RV32 SHA-256 three-row GC: tables={tables}, table_bytes={table_bytes}, touched_stack_slots={touched_stack_slots}, return_stack_slots={return_stack_slots}, garbling={garbling_elapsed:?}, evaluation={evaluation_elapsed:?}"
    );
}
