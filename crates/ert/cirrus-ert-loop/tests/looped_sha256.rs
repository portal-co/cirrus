//! The looped-circuit emulator driving the real RV64 SHA-256 self-test
//! workload on the host: the guest image is built by cargo (the same binary
//! the QEMU gate runs), and the interpreted hash must match the native one.
//! The workload's control flow is concrete, so this exercises the
//! single-candidate path end to end.

use std::{
    array,
    convert::Infallible,
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::OnceLock,
};

use cirrus_ert::{DefaultHandler, RawMemory, RvDefaultHandler};
use cirrus_ert_loop::LoopedMachine;
use cirrus_ert_sha256_fixture::sha256_compress;
use cirrus_volar_boolar::MuxTreeContext;

const TARGET: &str = "riscv64gc-unknown-none-elf";
const BASE: u64 = 0x8000_0000;
const INPUT: [u64; 16] = [0x6162_6380, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24];

static SELF_TEST_IMAGE: OnceLock<PathBuf> = OnceLock::new();

#[test]
fn looped_rv64_sha256_workload_matches_native() {
    let image = self_test_image();
    let elf = fs::read(image).expect("built ELF must be readable");
    let mapping = mapped_image(
        &elf,
        &[".text.ert_workload", ".rodata.ert_workload", ".rodata"],
    );
    let entry = symbol_address(&elf, "__ert_workload_entry");
    let memory = unsafe { RawMemory::new(mapping.as_ptr().wrapping_sub(BASE as usize), None) };

    let mut handler = RvDefaultHandler {
        inner: DefaultHandler {
            context: MuxTreeContext::new(()),
            hash: no_hash_mux,
        },
    };
    let mut vstack = [false; 65_536];
    let storage_bits = vstack.len();
    let mut rstack = [0u64; 256];
    let mut candidates = [0u64; 16];
    let args = INPUT.map(|input| (word(input), None));
    let mut machine = LoopedMachine::<_, bool, Infallible, 64, u64>::new(
        &mut handler,
        &mut vstack,
        storage_bits,
        memory,
        &mut rstack,
        entry,
        args,
        &mut candidates,
        &[],
        false,
        true,
    )
    .expect("machine");
    assert!(machine.run(100_000, |wire| *wire).unwrap());
    let results: [([bool; 64], Option<u64>); 2] = machine.results().unwrap();
    let expected = sha256_compress(
        INPUT[0] as u32,
        INPUT[1] as u32,
        INPUT[2] as u32,
        INPUT[3] as u32,
        INPUT[4] as u32,
        INPUT[5] as u32,
        INPUT[6] as u32,
        INPUT[7] as u32,
        INPUT[8] as u32,
        INPUT[9] as u32,
        INPUT[10] as u32,
        INPUT[11] as u32,
        INPUT[12] as u32,
        INPUT[13] as u32,
        INPUT[14] as u32,
        INPUT[15] as u32,
    );
    // RV64's ABI leaves the high bits of a 32-bit return value undefined.
    // The low word must match the native result.
    assert_eq!(value(&results[1].0) & 0xffff_ffff, u64::from(expected));
}

// --- Helpers ---

fn word(value: u64) -> [bool; 64] {
    array::from_fn(|bit| (value >> bit) & 1 != 0)
}

fn value(word: &[bool; 64]) -> u64 {
    word.iter()
        .enumerate()
        .fold(0, |v, (bit, set)| v | ((*set as u64) << bit))
}

fn no_hash_mux(_: &mut MuxTreeContext<()>, _: &[[bool; 64]]) -> Result<[u8; 32], Infallible> {
    Ok([0; 32])
}

// --- Bare-metal image builder ---

fn self_test_image() -> &'static Path {
    SELF_TEST_IMAGE
        .get_or_init(|| {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(|path| path.parent())
                .expect("cirrus-ert-loop lives below the workspace crates directory")
                .to_owned();
            let target_dir = root.join("target/cirrus-ert64-selftest");
            let build = Command::new("cargo")
                .current_dir(&root)
                .args([
                    "build",
                    "-p",
                    "cirrus-ert64-selftest",
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
                .expect("cargo must be available to build the bare-metal self-test");
            assert_success("building the RV64 self-test", &build);
            target_dir
                .join(TARGET)
                .join("release")
                .join("cirrus-ert64-selftest")
        })
        .as_path()
}

// --- Minimal ELF64 parsing (the workload's sections and one symbol). ---

fn elf_u16(image: &[u8], offset: usize) -> usize {
    u16::from_le_bytes(image[offset..offset + 2].try_into().unwrap()) as usize
}

fn elf_u32(image: &[u8], offset: usize) -> usize {
    u32::from_le_bytes(image[offset..offset + 4].try_into().unwrap()) as usize
}

fn elf_u64(image: &[u8], offset: usize) -> usize {
    u64::from_le_bytes(image[offset..offset + 8].try_into().unwrap()) as usize
}

fn elf_header(image: &[u8]) -> (usize, usize, usize, usize) {
    assert_eq!(&image[..4], b"\x7fELF", "image is an ELF file");
    assert_eq!(image[4], 2, "self-test ELF is 64-bit");
    assert_eq!(image[5], 1, "self-test ELF is little-endian");
    (
        elf_u64(image, 40),
        elf_u16(image, 58),
        elf_u16(image, 60),
        elf_u16(image, 62),
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
    let strings_offset = elf_u64(image, strings + 24);
    let strings_size = elf_u64(image, strings + 32);
    let start = strings_offset + elf_u32(image, header);
    let bytes = &image[start..strings_offset + strings_size];
    let length = bytes
        .iter()
        .position(|byte| *byte == 0)
        .expect("ELF section name is terminated");
    core::str::from_utf8(&bytes[..length]).expect("ELF section name is UTF-8")
}

fn selected_section(image: &[u8], wanted: &str) -> Option<(usize, usize, usize)> {
    let (_, _, count, _) = elf_header(image);
    for index in 0..count {
        let header = section_header(image, index);
        if section_name(image, header) == wanted {
            return Some((
                elf_u64(image, header + 16),
                elf_u64(image, header + 24),
                elf_u64(image, header + 32),
            ));
        }
    }
    None
}

fn mapped_image(image: &[u8], names: &[&str]) -> Vec<u8> {
    let selected: Vec<_> = names
        .iter()
        .filter_map(|name| selected_section(image, name))
        .collect();
    let base = BASE as usize;
    assert!(selected.iter().all(|(address, _, _)| *address >= base));
    let end = selected
        .iter()
        .map(|(address, _, size)| address + size)
        .max()
        .expect("at least one ELF section is mapped");
    let mut mapping = vec![0; end - base];
    for (address, offset, size) in selected {
        mapping[address - base..address - base + size]
            .copy_from_slice(&image[offset..offset + size]);
    }
    mapping
}

fn symbol_address(image: &[u8], wanted: &str) -> u64 {
    const SYMBOL_TABLE: usize = 2;
    let (_, _, count, _) = elf_header(image);
    for index in 0..count {
        let header = section_header(image, index);
        if elf_u32(image, header + 4) != SYMBOL_TABLE {
            continue;
        }
        let strings = section_header(image, elf_u32(image, header + 40));
        let strings_offset = elf_u64(image, strings + 24);
        let strings_size = elf_u64(image, strings + 32);
        let symbols_offset = elf_u64(image, header + 24);
        let symbols_size = elf_u64(image, header + 32);
        let entry_size = elf_u64(image, header + 56);
        assert_eq!(entry_size, 24, "self-test uses ELF64 symbol entries");
        for entry in (symbols_offset..symbols_offset + symbols_size).step_by(entry_size) {
            let name_offset = strings_offset + elf_u32(image, entry);
            let name_bytes = &image[name_offset..strings_offset + strings_size];
            let length = name_bytes
                .iter()
                .position(|byte| *byte == 0)
                .expect("symbol name is terminated");
            if core::str::from_utf8(&name_bytes[..length]).expect("symbol name is UTF-8") == wanted
            {
                return elf_u64(image, entry + 8) as u64;
            }
        }
    }
    panic!("ELF image lacks required symbol {wanted}");
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
