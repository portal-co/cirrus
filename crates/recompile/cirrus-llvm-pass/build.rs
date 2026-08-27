use std::env;
use std::path::PathBuf;

use volar_llvm_pass_build::{
    AutoRegister, PluginPass, ShimConfig, compile_and_link_shim, generate_shim_source, probe,
};

fn main() {
    let llvm = probe("CIRRUS_LLVM_CONFIG", "LLVM_SYS_221_PREFIX", "22");

    let config = ShimConfig {
        plugin_name: "cirrus-llvm-pass".into(),
        plugin_version: "0.1".into(),
        link_anchor_symbol: "cirrus_llvm_pass_link_anchor".into(),
        free_error_symbol: "cirrus_llvm_pass_free_error".into(),
        passes: vec![
            PluginPass {
                pipeline_name: "cirrus-lower".into(),
                run_symbol: "cirrus_llvm_pass_run".into(),
                error_prefix: "cirrus-lower".into(),
                // Full-LTO pre-link sees individual modules; its merged-
                // module hook (the FullLinkTimeOptimizationEarlyEP
                // registration every `AutoRegister` also gets) is the only
                // phase allowed to resolve cross-module selectors.
                auto_register: Some(AutoRegister {
                    skip_full_lto_prelink: true,
                }),
            },
            PluginPass {
                pipeline_name: "cirrus-deloopify".into(),
                run_symbol: "cirrus_llvm_pass_deloopify_run".into(),
                error_prefix: "cirrus-deloopify".into(),
                // Deliberately registered only under its own pipeline name,
                // not auto-run during ordinary/LTO compiles -- it never
                // runs unless a caller explicitly asks for it (e.g.
                // `-passes=cirrus-deloopify,cirrus-lower`).
                auto_register: None,
            },
        ],
    };
    let source = generate_shim_source(&config);

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    compile_and_link_shim(&source, &out_dir, &llvm.includedir, "cirrus_llvm_pass_shim");
}
