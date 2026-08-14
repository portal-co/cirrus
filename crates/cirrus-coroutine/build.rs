use std::env;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(cirrus_supported_target)");

    let target = env::var("TARGET").expect("Cargo must set TARGET");
    let supported = matches!(
        target.as_str(),
        "riscv32im-unknown-none-elf"
            | "thumbv8m.main-none-eabi"
            | "riscv64gc-unknown-none-elf"
            | "aarch64-unknown-none"
    ) || target.starts_with("aarch64-");

    if supported {
        println!("cargo::rustc-cfg=cirrus_supported_target");
    }
}
