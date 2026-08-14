use std::env;
use std::path::PathBuf;
use std::process::Command;

fn query(llvm_config: &str, argument: &str) -> String {
    let output = Command::new(llvm_config)
        .arg(argument)
        .output()
        .unwrap_or_else(|error| panic!("run {llvm_config} {argument}: {error}"));
    if !output.status.success() {
        panic!(
            "{llvm_config} {argument} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8(output.stdout)
        .expect("llvm-config output is UTF-8")
        .trim()
        .to_owned()
}

fn main() {
    println!("cargo:rerun-if-env-changed=CIRRUS_LLVM_CONFIG");
    println!("cargo:rerun-if-env-changed=LLVM_SYS_221_PREFIX");
    println!("cargo:rerun-if-changed=cxx/pass_shim.cpp");

    let llvm_config = env::var("CIRRUS_LLVM_CONFIG").unwrap_or_else(|_| {
        env::var("LLVM_SYS_221_PREFIX")
            .map(|prefix| {
                PathBuf::from(prefix)
                    .join("bin/llvm-config")
                    .to_string_lossy()
                    .into_owned()
            })
            .unwrap_or_else(|_| {
                [
                    "llvm-config".to_owned(),
                    "llvm-config-22".to_owned(),
                    "/opt/homebrew/opt/llvm@22/bin/llvm-config".to_owned(),
                    "/usr/lib/llvm-22/bin/llvm-config".to_owned(),
                ]
                .into_iter()
                .find(|candidate| Command::new(candidate).arg("--version").output().is_ok())
                .unwrap_or_else(|| "llvm-config".to_owned())
            })
    });
    let version = query(&llvm_config, "--version");
    assert!(
        version.starts_with("22."),
        "cirrus-llvm-pass requires LLVM 22, but {llvm_config} reports {version}"
    );
    let includedir = query(&llvm_config, "--includedir");

    cc::Build::new()
        .cpp(true)
        .file("cxx/pass_shim.cpp")
        .include(includedir)
        .flag_if_supported("-std=c++17")
        .warnings(false)
        .compile("cirrus_llvm_pass_shim");

    // Cargo links `cc` output as a static archive. The plugin's sole public
    // entrypoint lives in that archive, so ordinary archive extraction would
    // discard it before Clang has a chance to discover it.
    let archive = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"))
        .join("libcirrus_llvm_pass_shim.a");
    if cfg!(target_os = "macos") {
        println!(
            "cargo:rustc-cdylib-link-arg=-Wl,-force_load,{}",
            archive.display()
        );
        println!("cargo:rustc-cdylib-link-arg=-Wl,-exported_symbol,_llvmGetPassPluginInfo");
    } else {
        println!(
            "cargo:rustc-cdylib-link-arg=-Wl,--whole-archive,{},--no-whole-archive",
            archive.display()
        );
        println!("cargo:rustc-cdylib-link-arg=-Wl,--export-dynamic-symbol=llvmGetPassPluginInfo");
    }
}
