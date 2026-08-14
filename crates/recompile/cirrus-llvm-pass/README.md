# Cirrus LLVM pass

`cirrus-llvm-pass` is an LLVM 22 New-PM pass plugin. It lowers selected,
descriptor-backed LLVM functions with `cirrus-llvm-frontend`, prepares the
result, and emits a companion in the same module. The source function remains
ordinary native code.

The companion is named `<output-prefix><target-name>` (default
`__cirrus_`) and has this ABI:

```c
void companion(void *backend, void *scratch_buffer);
```

It calls the existing untagged `cirrus_rt_create`, `cirrus_rt_bitand`,
`cirrus_rt_bitor`, `cirrus_rt_bitxor`, and `cirrus_rt_mux` imports. Beside it,
the pass emits `<companion>_abi`, `<companion>_inputs`, and
`<companion>_outputs`. `CirrusProgramAbiV1` in
`cirrus-llvm-pass-api/include/cirrus_llvm_pass.h` describes those symbols and
the caller-owned scratch buffer.

## Source contract

The C header and the `no_std` `cirrus-llvm-pass-api` crate define the exact
V1 descriptor records. A descriptor is static metadata only: the pass reads
LLVM constant initializers and generated code never dereferences it.

Use one of these selection mechanisms:

```c
/* Select one direct target. The pass validates and removes this call. */
CIRRUS_ENTRY_MARKER(my_cirrus_marker, kernel, &kernel_descriptor)

/* Or retain a module-wide policy. Selector flags can be EXACT or PREFIX. */
CIRRUS_MODULE_CONFIG(((CirrusModuleConfigV1) {
    CIRRUS_ABI_VERSION_V1, 0, "kernel_", "__cirrus_",
    selectors, selector_count, 0,
}))
```

Rust has matching `cirrus_entry_marker!` and `cirrus_module_config!` macros.
They use `#[used]` so LLVM retains the marker/configuration in `llvm.used`.
`__cirrus_entry` is deliberately a normal external pseudo-intrinsic name;
there is no `llvm.cirrus.*` declaration to trigger LLVM's unknown-intrinsic
diagnostics.

Selection is deterministic: exact selectors, direct markers, and prefix
selectors are merged by target name. Repeating an identical selection is
deduplicated; conflicting descriptors or generated-symbol collisions are
diagnostics. Selected target functions must have a definition. A selected
kernel may call supported LLVM intrinsics and defined direct callees, but an
unresolved host import is rejected because pass plugins do not accept a dynamic
host-call registry.

## Building and loading

The plugin requires LLVM 22. Set `CIRRUS_LLVM_CONFIG` to the selected
`llvm-config` path, or set `LLVM_SYS_221_PREFIX` to the LLVM 22 prefix:

```sh
CIRRUS_LLVM_CONFIG=/path/to/llvm-config cargo build -p cirrus-llvm-pass
```

Load it during ordinary Clang optimization:

```sh
clang -O2 -fpass-plugin=/path/to/libcirrus_llvm_pass.dylib kernel.c -c -o kernel.o
```

For LTO, load the plugin in the linker. LLVM 22's LLD spelling is:

```sh
clang -O2 -flto -fuse-ld=lld \
  -Wl,--load-pass-plugin=/path/to/libcirrus_llvm_pass.dylib \
  inputs.o -o host
```

Some toolchain wrappers expose the equivalent option through `-mllvm`; use
their documented spelling (for example
`-Wl,-mllvm,-load-pass-plugin=/path/to/libcirrus_llvm_pass.dylib`) only when
that linker accepts it. Apple `ld` does not load New-PM pass plugins; use an
LLVM 22 LLD linker for LTO.

Full LTO runs `cirrus-lower` after modules merge, so a marker/selector in one
translation unit may target a definition in another. ThinLTO lowers before
cross-module resolution: the descriptor and selected definition must be
co-located, while the generated companion itself remains callable from other
translation units.

For explicit IR pipelines, use:

```sh
opt -load-pass-plugin=/path/to/libcirrus_llvm_pass.dylib \
  -passes=cirrus-lower input.ll -o output.bc
```

External integration tests discover compatible LLVM 22 Clang/`opt`, LLD, and
Rust toolchains automatically. CI should set
`CIRRUS_REQUIRE_EXTERNAL_LLVM_TESTS=1` so an unavailable LTO toolchain is a
failure rather than a skipped test.
