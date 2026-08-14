use std::{
    env,
    path::PathBuf,
    process::{Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

const PACKAGE: &str = "cirrus-coroutine-selftest";

#[test]
fn rv32im_runs_under_qemu() {
    run("riscv32im-unknown-none-elf");
}

#[test]
fn thumbv8m_runs_under_qemu() {
    run("thumbv8m.main-none-eabi");
}

#[test]
fn rv64gc_floating_point_runs_under_qemu() {
    run("riscv64gc-unknown-none-elf");
}

fn run(target: &str) {
    ensure_target(target);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("cirrus-coroutine lives below the workspace crates directory")
        .to_owned();
    let target_dir = root.join("target").join("cirrus-coroutine-selftest");
    let build = Command::new("cargo")
        .current_dir(&root)
        .args([
            "build",
            "-p",
            PACKAGE,
            "--features",
            "bare-metal",
            "--target",
            target,
            "--release",
            "--target-dir",
        ])
        .arg(&target_dir)
        .env("RUSTFLAGS", "-C panic=abort")
        .output()
        .expect("cargo must build the coroutine self-test");
    assert_success("building the coroutine self-test", &build);

    let image = target_dir.join(target).join("release").join(PACKAGE);
    let output = run_qemu(target, &image);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let passed = output.status.success()
        || (target == "thumbv8m.main-none-eabi" && text.contains("coroutine=00000000"));
    assert!(
        passed,
        "QEMU coroutine self-test for {target} failed with {}\nstdout:\n{}\nstderr:\n{}",
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
    assert_success("listing installed Rust targets", &installed);
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
    assert_success("installing the Rust target", &install);
}

fn run_qemu(target: &str, image: &PathBuf) -> Output {
    let mut command = match target {
        "riscv32im-unknown-none-elf" => {
            let mut command = Command::new("qemu-system-riscv32");
            command.args([
                "-machine", "virt", "-cpu", "rv32", "-m", "128M", "-bios", "none",
            ]);
            command
        }
        "riscv64gc-unknown-none-elf" => {
            let mut command = Command::new("qemu-system-riscv64");
            command.args([
                "-machine", "virt", "-cpu", "rv64", "-m", "128M", "-bios", "none",
            ]);
            command
        }
        "thumbv8m.main-none-eabi" => {
            let mut command = Command::new("qemu-system-arm");
            command.args([
                "-M",
                "mps2-an505",
                "-cpu",
                "cortex-m33",
                "-semihosting-config",
                "enable=on,target=native",
            ]);
            command
        }
        _ => unreachable!("target was checked before invoking QEMU"),
    };
    let mut child = command
        .arg("-kernel")
        .arg(image)
        .args(["-nographic", "-no-reboot"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the target-specific QEMU system binary is required");

    let deadline = Instant::now() + Duration::from_secs(30);
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
            child.kill().expect("timed-out QEMU must be killable");
            let output = child
                .wait_with_output()
                .expect("QEMU output must remain readable after timeout");
            panic!(
                "QEMU coroutine self-test for {target} timed out\nstdout:\n{}\nstderr:\n{}",
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
