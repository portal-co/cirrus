use std::{
    env,
    path::PathBuf,
    process::{Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

const TARGET: &str = "riscv32im-unknown-none-elf";

#[test]
fn rv32im_sha256_self_test_runs_under_qemu() {
    ensure_target_is_installed();

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("cirrus-ert lives in the workspace crates directory")
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

    let image = target_dir
        .join(TARGET)
        .join("release")
        .join("cirrus-ert-selftest");
    let output = run_qemu(&image);
    assert_success("running the RV32IM self-test under QEMU", &output);
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

fn run_qemu(image: &PathBuf) -> Output {
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
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if child
            .try_wait()
            .expect("QEMU process state must be observable")
            .is_some()
        {
            return child
                .wait_with_output()
                .expect("QEMU output must remain readable after exit");
        }
        if Instant::now() >= deadline {
            child
                .kill()
                .expect("timed-out QEMU process must be killable");
            let output = child
                .wait_with_output()
                .expect("QEMU output must remain readable after timeout");
            panic!(
                "RV32IM self-test timed out after 120 seconds\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
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
        String::from_utf8_lossy(&output.stderr),
    );
}
