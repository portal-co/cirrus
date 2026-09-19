fn main() {
    println!("cargo:rustc-link-arg=-T{}", concat!(env!("CARGO_MANIFEST_DIR"), "/linker.ld"));
    println!("cargo:rustc-link-arg=-nostdlib");
    println!("cargo:rustc-link-arg=-static");
    println!("cargo:rerun-if-changed=linker.ld");
}
