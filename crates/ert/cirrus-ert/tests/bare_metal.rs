use std::{
    array,
    convert::Infallible,
    env, fs,
    io::Read,
    os::unix::process::ExitStatusExt,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::{Mutex, OnceLock},
    thread,
    time::{Duration, Instant},
};

use cirrus_ert::{DefaultHandler, ErtError, RawMemory, RvDefaultHandler, ert_func};
use cirrus_ert_sha256_fixture::sha256_compress;

const TARGET: &str = "riscv32im-unknown-none-elf";
const BASE: u32 = 0x8000_0000;
const RUNNER_MEMORY_BYTES: usize = 128 * 1024 * 1024;
const BOOLAR_HEAP_BYTES: usize = 120 * 1024 * 1024;
const INPUT: [u32; 16] = [0x6162_6380, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24];

// Both tests build the same RV32 image and use process-wide resource counters.
// Serializing them keeps the CPU deltas attributable to the operation being
// reported rather than to a concurrently-running sibling test.
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());
static SELF_TEST_IMAGE: OnceLock<PathBuf> = OnceLock::new();

#[test]
fn rv32im_sha256_self_test_runs_under_qemu() {
    let _measurement = MEASUREMENT_LOCK
        .lock()
        .expect("measurement lock is not poisoned");
    let image = self_test_image();
    let image_bytes = allocated_image_bytes(&fs::read(image).expect("built ELF is readable"));
    assert!(
        image_bytes <= RUNNER_MEMORY_BYTES,
        "RV32 image uses {image_bytes} bytes, beyond the {RUNNER_MEMORY_BYTES}-byte QEMU RAM"
    );

    let run = run_qemu(image);
    assert_success("running the RV32IM self-test under QEMU", &run.output);
    let (heap_capacity, heap_used) = guest_heap_metrics(&run.output);
    assert_eq!(
        heap_capacity, BOOLAR_HEAP_BYTES,
        "the guest must report the large heap reserve configured for this test"
    );
    assert!(
        heap_used > 0 && heap_used <= heap_capacity,
        "guest bump allocation high-water mark {heap_used} must fit its {heap_capacity}-byte heap"
    );
    eprintln!(
        "RV32 QEMU self-test: runner_ram={} bytes, image={} bytes, remaining={} bytes, guest_heap_capacity={} bytes, guest_heap_used={} bytes, wall={:?}, cpu_user={:?}, cpu_system={:?}, child_peak_rss={} bytes",
        RUNNER_MEMORY_BYTES,
        image_bytes,
        RUNNER_MEMORY_BYTES - image_bytes,
        heap_capacity,
        heap_used,
        run.wall,
        run.cpu_user,
        run.cpu_system,
        run.peak_rss_bytes,
    );
}

#[test]
fn rv32im_sha256_workload_runs_on_host_with_measurements() {
    let _measurement = MEASUREMENT_LOCK
        .lock()
        .expect("measurement lock is not poisoned");
    run_host_image(self_test_image());
}

fn self_test_image() -> &'static Path {
    SELF_TEST_IMAGE
        .get_or_init(|| {
            ensure_target_is_installed();
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(|path| path.parent())
                .and_then(|path| path.parent())
                .expect("cirrus-ert lives below the workspace crates directory")
                .to_owned();
            let target_dir = root.join("target/cirrus-ert-selftest");
            let build = Command::new("cargo")
                .current_dir(&root)
                .args([
                    "build",
                    "-p",
                    "cirrus-ert-selftest",
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
            assert_success("building the RV32IM self-test", &build);
            target_dir
                .join(TARGET)
                .join("release")
                .join("cirrus-ert-selftest")
        })
        .as_path()
}

fn ensure_target_is_installed() {
    let installed = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .expect("rustup is required to install the RV32IM target");
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
        .expect("rustup is required to install the RV32IM target");
    assert_success("installing the RV32IM Rust target", &install);
}

struct MeasuredQemuRun {
    output: Output,
    wall: Duration,
    cpu_user: Duration,
    cpu_system: Duration,
    peak_rss_bytes: u64,
}

fn run_qemu(image: &Path) -> MeasuredQemuRun {
    let started = Instant::now();
    let mut child = Command::new("qemu-system-riscv32")
        .args([
            "-machine", "virt", "-cpu", "rv32", "-m", "128M", "-bios", "none", "-kernel",
        ])
        .arg(image)
        .args(["-nographic", "-no-reboot"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("qemu-system-riscv32 is required to run the bare-metal self-test");
    let pid = child.id().try_into().expect("QEMU PID fits a pid_t");
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut killed = false;
    let (output, usage) = loop {
        // SAFETY: `pid` is the child just spawned above; `wait4` is used with
        // WNOHANG until it exits, and initializes `status` and `usage` when it
        // returns that PID. Taking ownership of the child's pipes only occurs
        // after it has been reaped.
        let mut status = 0;
        let mut usage = unsafe { core::mem::zeroed::<libc::rusage>() };
        let waited = unsafe { libc::wait4(pid, &mut status, libc::WNOHANG, &mut usage) };
        if waited == pid {
            let mut stdout = Vec::new();
            child
                .stdout
                .take()
                .expect("QEMU stdout is piped")
                .read_to_end(&mut stdout)
                .expect("QEMU stdout remains readable after exit");
            let mut stderr = Vec::new();
            child
                .stderr
                .take()
                .expect("QEMU stderr is piped")
                .read_to_end(&mut stderr)
                .expect("QEMU stderr remains readable after exit");
            break (
                Output {
                    status: std::process::ExitStatus::from_raw(status),
                    stdout,
                    stderr,
                },
                ResourceUsage {
                    user: duration(usage.ru_utime),
                    system: duration(usage.ru_stime),
                    peak_rss_bytes: rss_bytes(usage.ru_maxrss),
                },
            );
        }
        assert_ne!(waited, -1, "QEMU wait4 must not fail");
        if Instant::now() >= deadline {
            if !killed {
                // SAFETY: `pid` still names the child because wait4 has not
                // reaped it; SIGKILL is used only after the fixed timeout.
                assert_eq!(
                    unsafe { libc::kill(pid, libc::SIGKILL) },
                    0,
                    "timed-out QEMU is killable"
                );
                killed = true;
            }
        }
        thread::sleep(Duration::from_millis(10));
    };
    if killed {
        panic!(
            "RV32IM self-test timed out after 120 seconds\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
    MeasuredQemuRun {
        output,
        wall: started.elapsed(),
        cpu_user: usage.user,
        cpu_system: usage.system,
        // `wait4` supplies this specific child's OS-recorded peak resident
        // size. macOS reports bytes while Linux reports KiB;
        // `resource_usage` normalizes those platform conventions.
        peak_rss_bytes: usage.peak_rss_bytes,
    }
}

fn run_host_image(image: &Path) {
    let elf = fs::read(image).expect("built RV32 ELF must be readable");
    let mapping = mapped_image(&elf, &[".text.ert_workload", ".rodata.ert_workload"]);
    let entry = symbol_address(&elf, "__ert_workload_entry");
    let memory = unsafe { RawMemory::new(mapping.as_ptr().wrapping_sub(BASE as usize), None) };
    let mut registers = [[false; 32]; 32];
    let mut constants = [None; 32];
    let mut rstack = [0; 256];
    let mut vstack = [false; 65_536];
    let args = INPUT.map(|value| (word(value), None));
    let before = resource_usage(libc::RUSAGE_SELF);
    let started = Instant::now();
    let result = ert_func::<_, _, 16, 2>(
        &mut RvDefaultHandler {
            inner: DefaultHandler {
                context: (),
                hash: no_hash,
            },
        },
        memory,
        &mut rstack,
        &mut vstack,
        entry,
        &mut registers,
        &mut constants,
        false,
        true,
        args,
    );
    let wall = started.elapsed();
    let after = resource_usage(libc::RUSAGE_SELF);
    let result = match result {
        Ok(result) => result,
        Err(ErtError::Decode(error)) => panic!("host RV32 image decode failed: {error:?}"),
        Err(ErtError::Unexpected) => panic!("host RV32 image violated the interpreter subset"),
        Err(ErtError::Emitted(error)) => match error {},
    };
    let expected = sha256_compress(
        INPUT[0], INPUT[1], INPUT[2], INPUT[3], INPUT[4], INPUT[5], INPUT[6], INPUT[7], INPUT[8],
        INPUT[9], INPUT[10], INPUT[11], INPUT[12], INPUT[13], INPUT[14], INPUT[15],
    );
    assert_eq!(result[0].1, Some(u32::MAX));
    assert_eq!(result[1], (word(expected), None));
    eprintln!(
        "RV32 host workload: wall={wall:?}, cpu_user={:?}, cpu_system={:?}, process_peak_rss={} bytes, value_stack_bytes={}, return_stack_bytes={}",
        after.user.saturating_sub(before.user),
        after.system.saturating_sub(before.system),
        after.peak_rss_bytes,
        core::mem::size_of_val(&vstack),
        core::mem::size_of_val(&rstack),
    );
}

fn word(value: u32) -> [bool; 32] {
    array::from_fn(|bit| value & (1 << bit) != 0)
}

fn no_hash(_: &mut (), _: &[[bool; 32]]) -> Result<[u8; 32], Infallible> {
    Ok([0; 32])
}

fn guest_heap_metrics(output: &Output) -> (usize, usize) {
    let output = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let value = |name: &str| {
        output
            .split_whitespace()
            .find_map(|field| field.strip_prefix(name))
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| panic!("QEMU output lacks {name} metric:\n{output}"))
    };
    (value("heap_capacity_bytes="), value("heap_used_bytes="))
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

fn allocated_image_bytes(image: &[u8]) -> usize {
    const SHF_ALLOC: usize = 0x2;
    let (_, _, count, _) = elf_header(image);
    let base = BASE as usize;
    (0..count)
        .map(|index| section_header(image, index))
        .filter(|header| elf_u32(image, *header + 8) & SHF_ALLOC != 0)
        .map(|header| elf_u32(image, header + 12) + elf_u32(image, header + 20))
        .max()
        .expect("ELF has an allocated section")
        .checked_sub(base)
        .expect("allocated RV32 image begins at the configured RAM base")
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

#[derive(Clone, Copy)]
struct ResourceUsage {
    user: Duration,
    system: Duration,
    peak_rss_bytes: u64,
}

fn resource_usage(who: libc::c_int) -> ResourceUsage {
    // SAFETY: `getrusage` initializes the supplied `rusage` on success and
    // the selected usage domain is one of the constants defined by libc.
    unsafe {
        let mut usage = core::mem::zeroed::<libc::rusage>();
        assert_eq!(
            libc::getrusage(who, &mut usage),
            0,
            "getrusage must succeed"
        );
        ResourceUsage {
            user: duration(usage.ru_utime),
            system: duration(usage.ru_stime),
            peak_rss_bytes: rss_bytes(usage.ru_maxrss),
        }
    }
}

fn duration(value: libc::timeval) -> Duration {
    let seconds = u64::try_from(value.tv_sec).expect("resource time is non-negative");
    let micros = u32::try_from(value.tv_usec).expect("resource time microseconds fit u32");
    Duration::new(seconds, micros * 1_000)
}

fn rss_bytes(value: libc::c_long) -> u64 {
    let value = u64::try_from(value).expect("peak RSS is non-negative");
    #[cfg(target_os = "macos")]
    {
        value
    }
    #[cfg(not(target_os = "macos"))]
    {
        value * 1024
    }
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
