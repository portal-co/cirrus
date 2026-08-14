use std::{
    array,
    convert::Infallible,
    env, fs,
    path::PathBuf,
    process::{Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use cirrus_armv8m_ert::{ErtError, RawMemory, ert_func};
use cirrus_ert_sha256_fixture::sha256_compress;

const TARGET: &str = "thumbv8m.main-none-eabi";

#[test]
fn thumbv8m_sha256_self_test_runs_under_qemu() {
    ensure_target();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .and_then(|path| path.parent())
        .unwrap()
        .to_owned();
    let target = root.join("target/cirrus-armv8m-ert-selftest");
    let build = Command::new("cargo")
        .current_dir(&root)
        .args([
            "build",
            "-p",
            "cirrus-armv8m-ert-selftest",
            "--features",
            "bare-metal",
            "--target",
            TARGET,
            "--release",
            "--target-dir",
        ])
        .arg(&target)
        .env("RUSTFLAGS", "-C panic=abort")
        .output()
        .expect("cargo must build the ARM bare-metal self-test");
    assert_success("building the Armv8-M self-test", &build);
    let image = target
        .join(TARGET)
        .join("release")
        .join("cirrus-armv8m-ert-selftest");
    run_host_image(&image);
    let output = run_qemu(&image);
    assert_qemu_success(&output);
}

fn run_host_image(image: &PathBuf) {
    const BASE: u32 = 0x2000_0000;
    let elf = fs::read(image).expect("built ELF must be readable");
    let word16 =
        |offset: usize| u16::from_le_bytes(elf[offset..offset + 2].try_into().unwrap()) as usize;
    let word32 =
        |offset: usize| u32::from_le_bytes(elf[offset..offset + 4].try_into().unwrap()) as usize;
    let sections = word32(32);
    let section_size = word16(46);
    let section_count = word16(48);
    let strings = word16(50);
    let string_header = sections + strings * section_size;
    let string_offset = word32(string_header + 16);
    let string_size = word32(string_header + 20);
    let names = &elf[string_offset..string_offset + string_size];
    let mut selected = Vec::new();
    for index in 0..section_count {
        let header = sections + index * section_size;
        let name = word32(header);
        let end = names[name..].iter().position(|byte| *byte == 0).unwrap();
        let name = std::str::from_utf8(&names[name..name + end]).unwrap();
        if matches!(name, ".text.ert_workload" | ".rodata.ert_workload") {
            selected.push((
                word32(header + 12),
                word32(header + 16),
                word32(header + 20),
            ));
        }
    }
    let end = selected
        .iter()
        .map(|(address, _, size)| address + size)
        .max()
        .expect("workload sections");
    let mut mapping = vec![0; end - BASE as usize];
    for (address, offset, size) in selected {
        mapping[address - BASE as usize..address - BASE as usize + size]
            .copy_from_slice(&elf[offset..offset + size]);
    }
    let input = [0x6162_6380, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24];
    // SAFETY: the wrapping base maps the high guest image range into the host
    // allocation; the test workload reads only this contiguous mapping.
    let memory = unsafe { RawMemory::new(mapping.as_ptr().wrapping_sub(BASE as usize), None) };
    let mut registers = [[false; 32]; 16];
    let mut constants = [None; 16];
    let mut rstack = [0; 512];
    let mut vstack = [false; 131_072];
    let args = input.map(|value| (array::from_fn(|bit| value & (1 << bit) != 0), None));
    let result = ert_func::<_, _, 16, 2>(
        &mut (),
        &mut |_| Ok::<_, Infallible>([0; 32]),
        memory,
        &mut rstack,
        &mut vstack,
        BASE | 1,
        &mut registers,
        &mut constants,
        false,
        true,
        args,
    );
    match result {
        Ok(result) => {
            let actual = result[1]
                .0
                .iter()
                .enumerate()
                .fold(0u32, |value, (bit, set)| value | ((*set as u32) << bit));
            let expected = sha256_compress(
                input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7],
                input[8], input[9], input[10], input[11], input[12], input[13], input[14],
                input[15],
            );
            assert_eq!(
                actual, expected,
                "host image result differs from native SHA-256 workload"
            );
            assert_eq!(result[1].1, None);
        }
        Err(ErtError::Decode(error)) => panic!("host image decode failed: {error:?}"),
        Err(ErtError::Unexpected) => panic!("host image violated the interpreter subset"),
        Err(ErtError::Emitted(error)) => match error {},
    }
}

fn ensure_target() {
    let installed = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .expect("rustup is required");
    assert_success("listing Rust targets", &installed);
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
        .expect("rustup must install the ARM target");
    assert_success("installing the Armv8-M Rust target", &install);
}

fn run_qemu(image: &PathBuf) -> Output {
    let mut child = Command::new("qemu-system-arm")
        .args([
            "-M",
            "mps2-an505",
            "-cpu",
            "cortex-m33",
            "-nographic",
            "-semihosting-config",
            "enable=on,target=native",
            "-kernel",
        ])
        .arg(image)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("qemu-system-arm is required to run the ARM self-test");
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if child
            .try_wait()
            .expect("QEMU process state must be observable")
            .is_some()
        {
            return child
                .wait_with_output()
                .expect("QEMU output must remain readable");
        }
        if Instant::now() >= deadline {
            child.kill().expect("timed-out QEMU must be killable");
            let output = child
                .wait_with_output()
                .expect("QEMU output must remain readable");
            panic!(
                "Armv8-M self-test timed out\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn assert_success(action: &str, output: &Output) {
    assert!(
        output.status.success(),
        "{action} failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_qemu_success(output: &Output) {
    // Some QEMU system-mode ARM builds report the semihosting application-exit
    // reason as a nonzero process status. The guest writes this marker only
    // after its native and symbolic SHA-256 comparisons have both succeeded.
    let completed =
        output.status.success() || String::from_utf8_lossy(&output.stderr).contains("ert=00000000");
    assert!(
        completed,
        "running the Armv8-M self-test under QEMU failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
