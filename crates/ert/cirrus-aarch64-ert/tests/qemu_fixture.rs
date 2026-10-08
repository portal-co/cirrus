use std::{
    env,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::OnceLock,
    time::{Duration, Instant},
};

const TARGET: &str = "aarch64-unknown-none";
const QEMU_TIMEOUT: Duration = Duration::from_secs(30);
static SELF_TEST_IMAGE: OnceLock<PathBuf> = OnceLock::new();

#[test]
#[ignore = "QEMU virt accepts the image but remains resident after the bare-metal SVC exit; needs a UART/semihosting exit protocol"]
fn aarch64_selftest_boots_and_exits_under_qemu() {
    let image = self_test_image();
    let started = Instant::now();
    let mut child = Command::new("qemu-system-aarch64")
        .args(["-machine", "virt", "-cpu", "max", "-m", "128M", "-kernel"])
        .arg(image)
        .args(["-display", "none", "-monitor", "none", "-no-reboot"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("qemu-system-aarch64 is required to run the AArch64 self-test");
    let deadline = Instant::now() + QEMU_TIMEOUT;
    let output = loop {
        match child
            .try_wait()
            .expect("QEMU child state must be queryable")
        {
            Some(status) => {
                let mut output = child
                    .wait_with_output()
                    .expect("QEMU output is collectable");
                output.status = status;
                break output;
            }
            None if Instant::now() >= deadline => {
                child.kill().expect("timed-out QEMU is killable");
                let output = child
                    .wait_with_output()
                    .expect("killed QEMU output is collectable");
                panic!(
                    "AArch64 self-test timed out after {:?}\nstdout:\n{}\nstderr:\n{}",
                    QEMU_TIMEOUT,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr),
                );
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    };
    assert!(
        output.status.success(),
        "AArch64 QEMU self-test failed after {:?} with status {:?}\nstdout:\n{}\nstderr:\n{}",
        started.elapsed(),
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
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
