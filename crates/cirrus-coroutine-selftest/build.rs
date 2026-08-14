use std::{env, path::PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let target = env::var("TARGET").expect("Cargo sets TARGET");
    let linker = match target.as_str() {
        "riscv32im-unknown-none-elf" | "riscv64gc-unknown-none-elf" => "linker-riscv.ld",
        "thumbv8m.main-none-eabi" => "linker-arm.ld",
        _ => panic!("unsupported self-test target: {target}"),
    };
    let linker = manifest.join(linker);
    println!("cargo::rerun-if-changed={}", linker.display());
    println!("cargo::rustc-link-arg=-T{}", linker.display());
}
