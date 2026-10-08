use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::OnceLock,
};

use cirrus_aarch64_ert::{Flow, RawMemory, decode, initial_state_with_arguments, step_with_hash};

const TARGET: &str = "aarch64-unknown-none";
const BASE: u64 = 0x4000_0000;
static SELF_TEST_IMAGE: OnceLock<PathBuf> = OnceLock::new();

#[test]
fn aarch64_selftest_host_fixture_decodes_and_exits() {
    let image = self_test_image();
    let elf = fs::read(image).expect("built AArch64 ELF must be readable");
    let mapping = mapped_image(&elf);
    let entry = symbol_address(&elf, "_start");
    let memory = unsafe { RawMemory::new(mapping.as_ptr().wrapping_sub(BASE as usize), None) };
    let mut state = initial_state_with_arguments(false, &true, 0x4800_0000, &[]).unwrap();
    let mut pc = entry;
    let mut hash_calls = 0u32;
    for _ in 0..4096 {
        let raw = u32::from_le_bytes(
            memory
                .read64::<4>(pc)
                .expect("mapped instruction fetch must stay in bounds"),
        );
        match step_with_hash(&mut (), &mut state, pc, raw, &false, &true, &mut |_, _| {
            hash_calls += 1;
            Ok([0x5a; 32])
        }) {
            Ok(Flow::Next(next)) => {
                let (Some(high), Some(low)) =
                    (concrete_bit(&next[32..]), concrete_bit(&next[..32]))
                else {
                    panic!("fixture next PC must be concrete at {pc:#x}");
                };
                pc = (u64::from(high) << 32) | u64::from(low);
            }
            Ok(Flow::Exit) => {
                assert_eq!(
                    hash_calls, 0,
                    "encoding fixture does not invoke the hash ABI"
                );
                assert_eq!(state.constants[0], Some(u64::MAX));
                assert!(state.done);
                return;
            }
            Err(cirrus_aarch64_ert::DecodeError::Unsupported(raw))
                if matches!(
                    decode(pc, raw),
                    Ok(cirrus_aarch64_ert::Instruction::Store { .. })
                ) =>
            {
                // The selftest's optional UART marker is outside the audited
                // ERT memory model. Continue to the next instruction.
                pc = pc.wrapping_add(4);
            }
            Err(error) => panic!("fixture failed at {pc:#x} raw={raw:#010x}: {error:?}"),
        }
    }
    panic!("fixture did not reach SVC exit within the step budget");
}

fn concrete_bit(bits: &[bool]) -> Option<u32> {
    let mut value = 0u32;
    for (index, bit) in bits.iter().enumerate() {
        value |= u32::from(*bit) << index;
    }
    Some(value)
}

fn self_test_image() -> &'static Path {
    SELF_TEST_IMAGE
        .get_or_init(|| {
            ensure_target_is_installed();
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(|path| path.parent())
                .and_then(|path| path.parent())
                .expect("cirrus-aarch64-ert lives below the workspace crates directory")
                .to_owned();
            let target_dir = root.join("target/cirrus-aarch64-ert-selftest");
            let build = Command::new("cargo")
                .current_dir(&root)
                .args([
                    "build",
                    "-p",
                    "cirrus-aarch64-ert-selftest",
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
            assert_success("building the AArch64 self-test", &build);
            target_dir
                .join(TARGET)
                .join("release")
                .join("cirrus-aarch64-ert-selftest")
        })
        .as_path()
}

fn ensure_target_is_installed() {
    let installed = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .expect("rustup is required to install the AArch64 target");
    assert_success("listing installed Rust targets", &installed);
    if installed
        .stdout
        .split(|byte| *byte == b'\n')
        .any(|target| target == TARGET.as_bytes())
    {
        return;
    }
    let install = Command::new("rustup")
        .args(["target", "add", TARGET])
        .output()
        .expect("rustup is required to install the AArch64 target");
    assert_success("installing the AArch64 Rust target", &install);
}

fn assert_success(operation: &str, output: &Output) {
    assert!(
        output.status.success(),
        "{operation} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn mapped_image(elf: &[u8]) -> Vec<u8> {
    assert_eq!(&elf[..4], b"\x7fELF", "ELF64 image magic");
    assert_eq!(elf[4], 2, "ELF64 image");
    assert_eq!(elf[5], 1, "little-endian ELF");
    assert_eq!(u16::from_le_bytes(elf[16..18].try_into().unwrap()), 2);
    assert_eq!(u16::from_le_bytes(elf[18..20].try_into().unwrap()), 183);
    let program_header_offset = u64::from_le_bytes(elf[32..40].try_into().unwrap()) as usize;
    let program_header_size = usize::from(u16::from_le_bytes(elf[54..56].try_into().unwrap()));
    let program_header_count = usize::from(u16::from_le_bytes(elf[56..58].try_into().unwrap()));
    let mut size = 0u64;
    for index in 0..program_header_count {
        let header = &elf[program_header_offset + index * program_header_size..][..56];
        let kind = u32::from_le_bytes(header[0..4].try_into().unwrap());
        if kind != 1 {
            continue;
        }
        let address = u64::from_le_bytes(header[16..24].try_into().unwrap());
        let memory_size = u64::from_le_bytes(header[40..48].try_into().unwrap());
        size = size.max(address.checked_sub(BASE).unwrap() + memory_size);
    }
    let mut image = vec![0; usize::try_from(size).expect("AArch64 image fits host memory")];
    for index in 0..program_header_count {
        let header = &elf[program_header_offset + index * program_header_size..][..56];
        if u32::from_le_bytes(header[0..4].try_into().unwrap()) != 1 {
            continue;
        }
        let offset = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        let address = u64::from_le_bytes(header[16..24].try_into().unwrap());
        let file_size = u64::from_le_bytes(header[32..40].try_into().unwrap()) as usize;
        let memory_size = u64::from_le_bytes(header[40..48].try_into().unwrap()) as usize;
        let start = usize::try_from(address.checked_sub(BASE).unwrap()).unwrap();
        image[start..start + memory_size].fill(0);
        image[start..start + file_size].copy_from_slice(&elf[offset..offset + file_size]);
    }
    image
}

fn symbol_address(elf: &[u8], wanted: &str) -> u64 {
    let section_offset = u64::from_le_bytes(elf[40..48].try_into().unwrap()) as usize;
    let section_size = usize::from(u16::from_le_bytes(elf[58..60].try_into().unwrap()));
    let section_count = usize::from(u16::from_le_bytes(elf[60..62].try_into().unwrap()));
    let names_index = usize::from(u16::from_le_bytes(elf[62..64].try_into().unwrap()));
    let names = &elf[section_offset + names_index * section_size..][..64];
    let names_offset = u64::from_le_bytes(names[24..32].try_into().unwrap()) as usize;
    for index in 0..section_count {
        let section = &elf[section_offset + index * section_size..][..64];
        if u32::from_le_bytes(section[4..8].try_into().unwrap()) != 2 {
            continue;
        }
        let link =
            usize::try_from(u32::from_le_bytes(section[40..44].try_into().unwrap())).unwrap();
        let strings = &elf[section_offset + link * section_size..][..64];
        let strings_offset = u64::from_le_bytes(strings[24..32].try_into().unwrap()) as usize;
        let strings_size = u64::from_le_bytes(strings[32..40].try_into().unwrap()) as usize;
        let table_offset = u64::from_le_bytes(section[24..32].try_into().unwrap()) as usize;
        let table_size = u64::from_le_bytes(section[32..40].try_into().unwrap()) as usize;
        for symbol in elf[table_offset..table_offset + table_size].chunks_exact(24) {
            let name = u32::from_le_bytes(symbol[0..4].try_into().unwrap()) as usize;
            let end = elf[strings_offset + name..strings_offset + strings_size]
                .iter()
                .position(|byte| *byte == 0)
                .expect("ELF symbol names are NUL-terminated");
            if &elf[strings_offset + name..strings_offset + name + end] == wanted.as_bytes() {
                let _ = names_offset;
                return u64::from_le_bytes(symbol[8..16].try_into().unwrap());
            }
        }
    }
    panic!("ELF symbol {wanted} not found");
}
