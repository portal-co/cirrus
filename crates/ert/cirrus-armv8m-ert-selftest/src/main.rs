#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec;
use core::{
    alloc::{GlobalAlloc, Layout},
    array,
    cell::UnsafeCell,
    convert::Infallible,
    panic::PanicInfo,
    ptr::{self, null_mut},
};

use cirrus_armv8m_ert::{
    ArmDefaultHandler, DecodeError, DefaultHandler, ErtError, RawMemory, SecurityAttribute,
    SecurityState, ert_func,
};
use cirrus_ert_sha256_fixture::sha256_compress;
use cirrus_volar_boolar::{MuxTreeContext, StorageBank, execute};
use volar_ir::{
    boolar::{BIrStmt, LaneId},
    circuit::BCircuit,
    ir::IRVarId,
};
use volar_ir_common::{Node, StorageId};

const BOOLAR_HEAP_BYTES: usize = 32 * 1024;

#[repr(align(16))]
struct Heap([u8; BOOLAR_HEAP_BYTES]);

struct BumpAllocator {
    cursor: UnsafeCell<usize>,
    heap: UnsafeCell<Heap>,
}

unsafe impl Sync for BumpAllocator {}

impl BumpAllocator {
    const fn new() -> Self {
        Self {
            cursor: UnsafeCell::new(0),
            heap: UnsafeCell::new(Heap([0; BOOLAR_HEAP_BYTES])),
        }
    }
}

unsafe impl GlobalAlloc for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let cursor = unsafe { *self.cursor.get() };
        let aligned = cursor
            .checked_add(layout.align() - 1)
            .map(|value| value & !(layout.align() - 1))
            .unwrap_or(BOOLAR_HEAP_BYTES);
        let Some(end) = aligned.checked_add(layout.size()) else {
            return null_mut();
        };
        if end > BOOLAR_HEAP_BYTES {
            return null_mut();
        }
        unsafe { *self.cursor.get() = end };
        let base = unsafe { (*self.heap.get()).0.as_mut_ptr() };
        unsafe { base.add(aligned) }
    }

    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}

#[global_allocator]
static ALLOCATOR: BumpAllocator = BumpAllocator::new();

core::arch::global_asm!(
    r#"
    .section .vector_table, "a"
    .globl __vector_table
__vector_table:
    .word _stack_top
    .word reset_handler
    .rept 14
    .word 0
    .endr

    .section .text.init, "ax"
    .thumb
    .thumb_func
    .globl reset_handler
reset_handler:
    ldr sp, =_stack_top
    ldr r0, =rust_main
    bx r0
1:  b 1b

    .section .text.ert_workload, "ax"
    .thumb
    .thumb_func
    .globl __ert_workload_entry
__ert_workload_entry:
    bl sha256_compress
    mov r1, r0
    movs r0, #0
    subs r0, #1
    svc #0
"#
);

unsafe extern "C" {
    static __ert_workload_entry: u8;
}

const INPUT: [u32; 16] = [0x6162_6380, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24];

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    finish(2)
}

#[unsafe(no_mangle)]
extern "C" fn rust_main() -> ! {
    if !boolar_storage_probe() {
        finish(0xc3);
    }
    let expected = sha256_compress(
        INPUT[0], INPUT[1], INPUT[2], INPUT[3], INPUT[4], INPUT[5], INPUT[6], INPUT[7], INPUT[8],
        INPUT[9], INPUT[10], INPUT[11], INPUT[12], INPUT[13], INPUT[14], INPUT[15],
    );
    let mut handler = ArmDefaultHandler {
        inner: DefaultHandler {
            context: MuxTreeContext::new(()),
            hash: no_hash,
        },
        svc_permitted: permit_all,
        security_attribute: always_secure,
    };
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    let mut rstack = [0; 512];
    let mut vstack = [false; 131_072];
    let storage_bits = vstack.len();
    let args = INPUT.map(|input| (word(input), None));
    let entry = ptr::addr_of!(__ert_workload_entry) as usize as u32 | 1;
    // SAFETY: QEMU maps all linked RAM image bytes at their guest addresses;
    // the interpreted workload fetches only code/static data in that image.
    let memory = unsafe { RawMemory::new(ptr::null(), None) };
    let results = match ert_func::<_, _, 16, 2, _>(
        &mut handler,
        &mut vstack,
        storage_bits,
        memory,
        &mut rstack,
        entry,
        &mut regs,
        &mut constants,
        false,
        true,
        args,
    ) {
        Ok(results) => results,
        Err(ErtError::Emitted(_)) => finish(0xe3),
        Err(ErtError::Decode(DecodeError::Unsupported(instruction))) => finish(instruction),
        Err(ErtError::Decode(DecodeError::Malformed(instruction))) => finish(instruction),
        Err(ErtError::Decode(DecodeError::Truncated)) => finish(0xe4),
        Err(ErtError::Unexpected) => finish(0xe5),
    };
    if results[0].1 != Some(u32::MAX) {
        finish(0xc4);
    }
    if results[1].0 != word(expected) {
        finish(0xc5);
    }
    if results[1].1.is_some() {
        finish(0xc6);
    }
    finish(0)
}

fn word(value: u32) -> [bool; 32] {
    array::from_fn(|bit| value & (1 << bit) != 0)
}
fn no_hash(_: &mut MuxTreeContext<()>, _: &[[bool; 32]]) -> Result<[u8; 32], Infallible> {
    Ok([0; 32])
}

fn boolar_storage_probe() -> bool {
    let circuit = BCircuit {
        params: 2,
        stmts: vec![
            Node::new(BIrStmt::One, (), None),
            Node::new(
                BIrStmt::StorageWrite {
                    storage: StorageId(0),
                    lane: LaneId(0),
                    src: IRVarId(0),
                    addr: vec![IRVarId(1), IRVarId(2)],
                },
                (),
                None,
            ),
            Node::new(
                BIrStmt::StorageRead {
                    storage: StorageId(0),
                    lane: LaneId(0),
                    addr: vec![IRVarId(1), IRVarId(2)],
                },
                (),
                None,
            ),
        ],
        pre_init: vec![],
        outputs: vec![IRVarId(4)],
    };
    let mut cells = [false; 4];
    let mut banks = [StorageBank {
        storage: StorageId(0),
        lane: LaneId(0),
        address_bits: 2,
        value: &mut cells[..],
    }];
    let mut context = MuxTreeContext::new(());
    matches!(
        execute(&mut context, &circuit, &[true, false], &mut banks),
        Ok(outputs) if outputs.as_slice() == [true] && cells == [false, false, true, false]
    )
}

fn permit_all<H>(_: &mut H, _: SecurityState) -> bool {
    true
}

fn always_secure<H>(_: &mut H, _: u32) -> SecurityAttribute {
    SecurityAttribute::Secure
}

fn finish(code: u32) -> ! {
    // QEMU's ARM semihosting SYS_EXIT_EXTENDED conveys a success/failure status
    // to the host test after the interpreter has exercised `SVC #0` internally.
    let mut report = *b"ert=00000000\n\0";
    for (shift, byte) in (0..8).rev().zip(&mut report[4..12]) {
        let digit = ((code >> (shift * 4)) & 15) as u8;
        *byte = if digit < 10 {
            b'0' + digit
        } else {
            b'a' + digit - 10
        };
    }
    unsafe {
        core::arch::asm!(
            "mov r0, #4",
            "mov r1, {message}",
            "bkpt 0xab",
            message = in(reg) report.as_ptr(),
            options(nostack),
        );
        if code == 0 {
            core::arch::asm!(
                "mov r0, #0x18",
                "mov r1, r2",
                "bkpt 0xab",
                in("r2") 0x20_026u32,
                options(noreturn)
            );
        }
        EXIT_ARGS[1] = code;
        core::arch::asm!(
            "mov r0, #0x20",
            "mov r1, {args}",
            "bkpt 0xab",
            args = in(reg) ptr::addr_of!(EXIT_ARGS),
            options(noreturn)
        );
    }
}

static mut EXIT_ARGS: [u32; 2] = [0x20_026, 0];
