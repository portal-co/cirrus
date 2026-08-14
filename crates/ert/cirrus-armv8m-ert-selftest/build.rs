use std::path::PathBuf;

fn main() {
    let linker = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("linker.ld");
    println!("cargo:rerun-if-changed={}", linker.display());
    println!("cargo:rustc-link-arg=-T{}", linker.display());
}
