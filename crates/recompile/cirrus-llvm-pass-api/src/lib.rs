#![no_std]
#![warn(missing_docs)]

//! Static source-side descriptors consumed by `cirrus-llvm-pass`.
//!
//! The pass reads these values from LLVM constant initializers; no descriptor
//! is dereferenced by generated code.  A source language should retain a
//! [`CirrusModuleConfigV1`] or [`CirrusEntryDescriptorV1`] through
//! `llvm.used` and use [`cirrus_entry_marker`] for direct marker selection.

use core::ffi::c_void;

/// The descriptor ABI version accepted by the first LLVM pass release.
pub const ABI_VERSION_V1: u32 = 1;

/// Select a descriptor explicitly, regardless of the source-name prefix.
pub const SELECT_EXACT: u32 = 1;
/// Select a descriptor when its target name begins the configured source prefix.
pub const SELECT_PREFIX: u32 = 2;

/// A scalar argument which is secret circuit input data.
pub const ARGUMENT_SYMBOLIC: u32 = 1;
/// A scalar argument whose `value` is concrete public data.
pub const ARGUMENT_CONCRETE: u32 = 2;
/// A concrete-address pointer argument backed by `bytes`.
pub const ARGUMENT_REGION: u32 = 3;

/// A public initial byte for a region.
pub const BYTE_CONCRETE: u8 = 1;
/// A symbolic input byte for a region.
pub const BYTE_SYMBOLIC: u8 = 2;

/// Export the selected function's scalar integer return lanes.
pub const EXPORT_RETURN: u32 = 1;
/// Export bytes from an entry pointer region.
pub const EXPORT_ARGUMENT_MEMORY: u32 = 2;
/// Export bytes from a named LLVM global.
pub const EXPORT_GLOBAL_MEMORY: u32 = 3;

/// One static byte classification in a pointer or global region.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CirrusRegionByteV1 {
    /// [`BYTE_CONCRETE`] or [`BYTE_SYMBOLIC`].
    pub kind: u8,
    /// The value when `kind` is [`BYTE_CONCRETE`].
    pub value: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 2],
}

/// One entry-function argument binding.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CirrusArgumentV1 {
    /// One of `ARGUMENT_*`.
    pub kind: u32,
    /// Nonzero permits stores into a region argument.
    pub writable: u32,
    /// Concrete scalar value for [`ARGUMENT_CONCRETE`].
    pub value: u64,
    /// Region byte classifications for [`ARGUMENT_REGION`].
    pub bytes: *const CirrusRegionByteV1,
    /// Number of elements in `bytes`.
    pub bytes_len: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
}

// These records are immutable compile-time metadata. Their raw pointers are
// only read by the LLVM pass while decoding constant initializers.
unsafe impl Sync for CirrusArgumentV1 {}

/// A named mutable LLVM global binding.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CirrusGlobalV1 {
    /// NUL-terminated unmangled LLVM global name, without `@`.
    pub name: *const u8,
    /// Nonzero permits stores into this global.
    pub writable: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// Initial byte classifications.
    pub bytes: *const CirrusRegionByteV1,
    /// Number of elements in `bytes`.
    pub bytes_len: u32,
    /// Reserved; must be zero.
    pub reserved2: u32,
}

unsafe impl Sync for CirrusGlobalV1 {}

/// One requested output slice.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CirrusExportV1 {
    /// One of `EXPORT_*`.
    pub kind: u32,
    /// Entry argument index for argument memory exports.
    pub argument: u32,
    /// NUL-terminated global name for global memory exports, otherwise null.
    pub name: *const u8,
    /// First byte in the selected region/global.
    pub offset: u32,
    /// Number of bytes for memory exports.
    pub len: u32,
}

unsafe impl Sync for CirrusExportV1 {}

/// Bounds for the bounded LLVM execution performed by the pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CirrusLoweringLimitsV1 {
    /// Maximum dynamic instruction visits.
    pub max_instructions: u64,
    /// Maximum direct calls.
    pub max_calls: u64,
    /// Maximum private alloca bytes.
    pub max_alloca_bytes: u64,
}

/// The full source-side lowering request for one target function.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CirrusEntryDescriptorV1 {
    /// Must equal [`ABI_VERSION_V1`].
    pub version: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// Entry argument bindings.
    pub arguments: *const CirrusArgumentV1,
    /// Number of elements in `arguments`.
    pub arguments_len: u32,
    /// Number of global bindings.
    pub globals_len: u32,
    /// Global bindings.
    pub globals: *const CirrusGlobalV1,
    /// Requested output lanes/ranges.
    pub exports: *const CirrusExportV1,
    /// Number of elements in `exports`.
    pub exports_len: u32,
    /// Reserved; must be zero.
    pub reserved2: u32,
    /// Bounded lowering limits.
    pub limits: CirrusLoweringLimitsV1,
}

unsafe impl Sync for CirrusEntryDescriptorV1 {}

/// A target/descriptor pair in an in-module selector table.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CirrusSelectorV1 {
    /// Function address. The pass requires a direct LLVM function reference.
    pub target: *const c_void,
    /// Static request for `target`.
    pub descriptor: *const CirrusEntryDescriptorV1,
    /// Bitwise union of [`SELECT_EXACT`] and [`SELECT_PREFIX`].
    pub flags: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
}

unsafe impl Sync for CirrusSelectorV1 {}

/// In-module selection and naming policy.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CirrusModuleConfigV1 {
    /// Must equal [`ABI_VERSION_V1`].
    pub version: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// NUL-terminated source prefix used with [`SELECT_PREFIX`].
    pub source_prefix: *const u8,
    /// NUL-terminated prefix for generated companion symbols.
    pub output_prefix: *const u8,
    /// Target/descriptor records.
    pub selectors: *const CirrusSelectorV1,
    /// Number of elements in `selectors`.
    pub selectors_len: u32,
    /// Reserved; must be zero.
    pub reserved2: u32,
}

unsafe impl Sync for CirrusModuleConfigV1 {}

/// Metadata exported beside every generated companion entrypoint.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CirrusProgramAbiV1 {
    /// ABI version, currently [`ABI_VERSION_V1`].
    pub version: u32,
    /// Reserved for future ABI flags; zero in V1.
    pub flags: u32,
    /// Scratch-buffer slots required by the companion.
    pub slots: u32,
    /// Number of input slot indices.
    pub input_count: u32,
    /// Number of output slot indices.
    pub output_count: u32,
    /// Slot indices populated by the caller before invocation.
    pub inputs: *const u32,
    /// Slot indices read by the caller after invocation.
    pub outputs: *const u32,
}

unsafe extern "C" {
    #[doc(hidden)]
    pub fn __cirrus_entry(target: *const c_void, descriptor: *const CirrusEntryDescriptorV1);
}

/// Retain and call a direct `__cirrus_entry` marker without exposing it at
/// runtime. The LLVM pass erases the marker function after it emits the
/// companion.
#[macro_export]
macro_rules! cirrus_entry_marker {
    ($name:ident, $target:path, $descriptor:expr) => {
        #[used]
        static $name: unsafe extern "C" fn() = {
            unsafe extern "C" fn marker() {
                // SAFETY: this is an intentional compile-time pseudo-intrinsic
                // consumed by cirrus-llvm-pass before final linking.
                unsafe {
                    $crate::__cirrus_entry(
                        $target as *const () as *const core::ffi::c_void,
                        $descriptor as *const $crate::CirrusEntryDescriptorV1,
                    );
                }
            }
            marker
        };
    };
}

/// Retain one module-level selection policy under the exact symbol name the
/// pass recognizes. Only one invocation is allowed per LLVM module.
#[macro_export]
macro_rules! cirrus_module_config {
    ($config:expr) => {
        #[used]
        #[unsafe(no_mangle)]
        static __cirrus_module_config: $crate::CirrusModuleConfigV1 = $config;
    };
}
