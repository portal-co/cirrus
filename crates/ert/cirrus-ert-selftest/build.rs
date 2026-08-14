use std::env;

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").expect("Cargo sets the manifest directory");
    println!("cargo::rerun-if-changed=linker.ld");
    println!("cargo::rustc-link-arg=-T{manifest}/linker.ld");
}
