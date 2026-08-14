//! End-to-end coverage for the dynamically loaded LLVM pass.
//!
//! These tests intentionally use installed tools rather than inkwell so the
//! plugin loading, ordinary compilation, Full/Thin LTO hooks, and Rust's
//! linker-plugin bitcode all exercise the real public boundary. Set
//! `CIRRUS_REQUIRE_EXTERNAL_LLVM_TESTS=1` in CI to turn an unavailable LLVM 22
//! compiler, LLD, or Rust 22 toolchain into a test failure.

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const REQUIRED: &str = "CIRRUS_REQUIRE_EXTERNAL_LLVM_TESTS";

struct Tools {
    clang: PathBuf,
    opt: PathBuf,
    lld: PathBuf,
    rustc: PathBuf,
    plugin: PathBuf,
    api_rlib: PathBuf,
    root: PathBuf,
}

#[test]
fn clang_and_rust_lto_load_the_pass_and_execute_companions() {
    let Some(tools) = Tools::discover() else {
        if env::var_os(REQUIRED).is_some() {
            panic!("{REQUIRED}=1 requires Clang, opt, LLD, and rustc built with LLVM 22");
        }
        eprintln!("skipping external Cirrus LLVM pass test: compatible LLVM 22 tools unavailable");
        return;
    };
    let output = env::temp_dir().join(format!("cirrus-llvm-pass-{}", std::process::id()));
    fs::create_dir_all(&output).expect("create test output directory");
    let fixtures = tools
        .root
        .join("crates/recompile/cirrus-llvm-pass/tests/fixtures");
    let header = tools
        .root
        .join("crates/recompile/cirrus-llvm-pass-api/include");

    let direct_object = output.join("direct.o");
    run(Command::new(&tools.clang)
        .arg("-O2")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg(format!("-I{}", header.display()))
        .arg(format!("-fpass-plugin={}", tools.plugin.display()))
        .arg("-c")
        .arg(fixtures.join("direct.c"))
        .arg("-o")
        .arg(&direct_object));
    assert_symbols(
        &tools,
        &direct_object,
        &["__cirrus_kernel", "__cirrus_kernel_abi"],
    );
    let direct = output.join("direct");
    link_plain(&tools, &direct, &[direct_object], &fixtures, &header);
    run(&mut Command::new(&direct));

    // Explicit `opt` registration has a separate public loading path from
    // Clang's optimizer extension point.
    let direct_ir = output.join("direct.ll");
    run(Command::new(&tools.clang)
        .arg("-O0")
        .arg("-S")
        .arg("-emit-llvm")
        .arg(format!("-I{}", header.display()))
        .arg(fixtures.join("direct.c"))
        .arg("-o")
        .arg(&direct_ir));
    let opt_ir = output.join("direct.opt.ll");
    run(Command::new(&tools.opt)
        .arg(format!("-load-pass-plugin={}", tools.plugin.display()))
        .arg("-passes=cirrus-lower")
        .arg("-S")
        .arg(&direct_ir)
        .arg("-o")
        .arg(&opt_ir));
    let opt_text = fs::read_to_string(&opt_ir).expect("read opt output");
    assert!(opt_text.contains("@__cirrus_kernel_abi"));
    assert!(!opt_text.contains("call void @__cirrus_entry"));

    let marker = output.join("lto-marker.o");
    let kernel = output.join("lto-kernel.o");
    compile_lto(
        &tools,
        "-flto",
        &fixtures.join("lto_marker.c"),
        &marker,
        &header,
    );
    compile_lto(
        &tools,
        "-flto",
        &fixtures.join("lto_kernel.c"),
        &kernel,
        &header,
    );
    let full_c = output.join("full-c");
    link_lto(
        &tools,
        "-flto",
        &full_c,
        &[marker, kernel],
        &fixtures,
        &header,
        &output,
    );
    run(&mut Command::new(&full_c));

    let thin_kernel = output.join("thin-kernel.o");
    compile_lto(
        &tools,
        "-flto=thin",
        &fixtures.join("direct.c"),
        &thin_kernel,
        &header,
    );
    let thin_c = output.join("thin-c");
    link_lto(
        &tools,
        "-flto=thin",
        &thin_c,
        &[thin_kernel],
        &fixtures,
        &header,
        &output,
    );
    run(&mut Command::new(&thin_c));

    let rust_archive = output.join("libcirrus_rust_kernel.a");
    run(Command::new(&tools.rustc)
        .arg("--edition=2024")
        .arg("--crate-type=staticlib")
        .arg("-O")
        .arg("-C")
        .arg("panic=abort")
        .arg("-C")
        .arg("linker-plugin-lto")
        .arg("--extern")
        .arg(format!("cirrus_llvm_pass_api={}", tools.api_rlib.display()))
        .arg(fixtures.join("rust_kernel.rs"))
        .arg("-o")
        .arg(&rust_archive));
    for (mode, name) in [("-flto", "full-rust"), ("-flto=thin", "thin-rust")] {
        let executable = output.join(name);
        link_rust_lto(
            &tools,
            mode,
            &executable,
            &rust_archive,
            &fixtures,
            &header,
            &output,
        );
        run(&mut Command::new(&executable));
    }
}

impl Tools {
    fn discover() -> Option<Self> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)?
            .to_path_buf();
        let llvm_bin = env::var_os("CIRRUS_LLVM_CONFIG")
            .map(PathBuf::from)
            .and_then(|path| path.parent().map(Path::to_path_buf));
        let clang = find_llvm_tool(
            "clang",
            llvm_bin.as_deref(),
            &["clang-22", "clang", "/opt/homebrew/opt/llvm@22/bin/clang"],
        )?;
        let opt = find_llvm_tool(
            "opt",
            llvm_bin.as_deref().or_else(|| clang.parent()),
            &["opt-22", "opt", "/opt/homebrew/opt/llvm@22/bin/opt"],
        )?;
        let lld = find_lld()?;
        let rustc = PathBuf::from(env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc")));
        if !version_contains(&rustc, &["-vV"], "LLVM version: 22.") {
            return None;
        }
        let target = cargo_target_dir(&root);
        // `cargo test` only guarantees an rlib for this package. Build the
        // cdylib explicitly so the plugin loaded below exactly matches the
        // source under test instead of a stale developer artifact.
        run(
            Command::new(env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo")))
                .current_dir(&root)
                .arg("build")
                .arg("--quiet")
                .arg("-p")
                .arg("cirrus-llvm-pass"),
        );
        let plugin = target
            .join("debug")
            .join(dynamic_library_name("cirrus_llvm_pass"));
        if !plugin.is_file() {
            return None;
        }
        let api_rlib = fs::read_dir(target.join("debug/deps"))
            .ok()?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with("libcirrus_llvm_pass_api-") && name.ends_with(".rlib")
                    })
            })?;
        Some(Self {
            clang,
            opt,
            lld,
            rustc,
            plugin,
            api_rlib,
            root,
        })
    }
}

fn cargo_target_dir(root: &Path) -> PathBuf {
    env::var_os("CARGO_TARGET_DIR").map_or_else(|| root.join("target"), PathBuf::from)
}

fn dynamic_library_name(stem: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else if cfg!(target_os = "windows") {
        format!("{stem}.dll")
    } else {
        format!("lib{stem}.so")
    }
}

fn find_llvm_tool(name: &str, preferred_bin: Option<&Path>, fallbacks: &[&str]) -> Option<PathBuf> {
    preferred_bin
        .map(|directory| directory.join(name))
        .into_iter()
        .chain(fallbacks.iter().map(PathBuf::from))
        .find(|candidate| version_contains(candidate, &["--version"], "22."))
}

fn find_lld() -> Option<PathBuf> {
    if let Some(path) = env::var_os("CIRRUS_LLD").map(PathBuf::from) {
        return version_contains(&path, &["--version"], "LLD 22.").then_some(path);
    }
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));
    let sysroot = command_text(Path::new(&rustc), &["--print", "sysroot"])?;
    let rustc_version = command_text(Path::new(&rustc), &["-vV"])?;
    let host = rustc_version
        .lines()
        .find_map(|line| line.strip_prefix("host: "))?;
    let candidates = if cfg!(target_os = "macos") {
        ["ld64.lld", "ld.lld"]
    } else {
        ["ld.lld", "ld64.lld"]
    };
    candidates
        .into_iter()
        .map(|name| {
            Path::new(sysroot.trim())
                .join("lib/rustlib")
                .join(host)
                .join("bin/gcc-ld")
                .join(name)
        })
        .find(|candidate| version_contains(candidate, &["--version"], "LLD 22."))
}

fn version_contains(program: &Path, arguments: &[&str], needle: &str) -> bool {
    command_text(program, arguments).is_some_and(|text| text.contains(needle))
}

fn command_text(program: &Path, arguments: &[&str]) -> Option<String> {
    let output = Command::new(program).args(arguments).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

fn compile_lto(tools: &Tools, mode: &str, source: &Path, output: &Path, header: &Path) {
    run(Command::new(&tools.clang)
        .arg("-O2")
        .arg(mode)
        .arg("-c")
        .arg(format!("-I{}", header.display()))
        .arg(source)
        .arg("-o")
        .arg(output));
}

fn link_plain(tools: &Tools, output: &Path, objects: &[PathBuf], fixtures: &Path, header: &Path) {
    let mut command = Command::new(&tools.clang);
    command.arg("-O2").arg(format!("-I{}", header.display()));
    command.args(objects);
    command
        .arg(fixtures.join("plaintext_runtime.c"))
        .arg(fixtures.join("direct_host.c"))
        .arg("-o")
        .arg(output);
    run(&mut command);
}

fn link_lto(
    tools: &Tools,
    mode: &str,
    output: &Path,
    objects: &[PathBuf],
    fixtures: &Path,
    header: &Path,
    work: &Path,
) {
    let mut command = lto_link_command(tools, mode, output, fixtures, header, work);
    command.args(objects);
    run(&mut command);
}

fn link_rust_lto(
    tools: &Tools,
    mode: &str,
    output: &Path,
    archive: &Path,
    fixtures: &Path,
    header: &Path,
    work: &Path,
) {
    let mut command = lto_link_command(tools, mode, output, fixtures, header, work);
    if cfg!(target_os = "macos") {
        command.arg(format!("-Wl,-force_load,{}", archive.display()));
    } else {
        command
            .arg("-Wl,--whole-archive")
            .arg(archive)
            .arg("-Wl,--no-whole-archive");
    }
    run(&mut command);
}

fn lto_link_command(
    tools: &Tools,
    mode: &str,
    output: &Path,
    fixtures: &Path,
    header: &Path,
    work: &Path,
) -> Command {
    let mut command = Command::new(&tools.clang);
    command
        .arg("-O2")
        .arg(mode)
        .arg(format!("-fuse-ld={}", tools.lld.display()))
        .arg(format!("-I{}", header.display()))
        .arg(format!("-Wl,--load-pass-plugin={}", tools.plugin.display()));
    // A Rust linker-plugin-LTO archive may contribute ThinLTO bitcode even
    // when the C side asks for Full LTO. ld64.lld requires its object path in
    // that mixed mode.
    if cfg!(target_os = "macos") {
        command.arg(format!("-Wl,-object_path_lto,{}", work.display()));
    }
    command
        .arg(fixtures.join("plaintext_runtime.c"))
        .arg(fixtures.join("direct_host.c"))
        .arg("-o")
        .arg(output);
    command
}

fn assert_symbols(tools: &Tools, object: &Path, names: &[&str]) {
    let llvm_nm = tools.clang.parent().unwrap().join("llvm-nm");
    let output = if llvm_nm.is_file() {
        command_output(Command::new(llvm_nm).arg(object))
    } else {
        command_output(Command::new("nm").arg(object))
    };
    let symbols = String::from_utf8_lossy(&output.stdout);
    for name in names {
        assert!(
            symbols.contains(name),
            "missing generated symbol {name}: {symbols}"
        );
    }
}

fn run(command: &mut Command) {
    let output = command_output(command);
    assert!(
        output.status.success(),
        "command failed ({:?}):\nstdout:\n{}\nstderr:\n{}",
        command,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn command_output(command: &mut Command) -> Output {
    command
        .output()
        .unwrap_or_else(|error| panic!("run {command:?}: {error}"))
}
