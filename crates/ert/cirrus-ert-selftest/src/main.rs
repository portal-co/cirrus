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

use cirrus_ert::{DefaultHandler, RawMemory, RvDefaultHandler, ert_func};
use cirrus_ert_sha256_fixture::sha256_compress;
use cirrus_volar_boolar::{StorageBank, execute};
use volar_ir::{
    boolar::{BIrStmt, LaneId},
    circuit::BCircuit,
    ir::IRVarId,
};
use volar_ir_common::{Node, StorageId};

/// The QEMU RV32 runner has 128 MiB of RAM.  Keep an intentionally large
/// allocator below that ceiling so the symbolic test exercises the same memory
/// envelope as the runner while leaving room for the image, stacks, and a
/// guard band.
const BOOLAR_HEAP_BYTES: usize = 120 * 1024 * 1024;

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

    /// High-water mark for the no-deallocation allocator. The self-test emits
    /// this after its Boolar probe so CI and host runs can record the actual
    /// allocation requirement rather than inferring it from the heap reserve.
    fn used(&self) -> usize {
        // SAFETY: the bare-metal self-test is single-threaded and only reads
        // the cursor after all allocations for the probe have completed.
        unsafe { *self.cursor.get() }
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
    .section .text.init, "ax"
    .globl _start
_start:
    la t0, __bss_start
    la t1, __bss_end
.Lclear_bss:
    bgeu t0, t1, .Lstart_rust
    sb zero, 0(t0)
    addi t0, t0, 1
    j .Lclear_bss
.Lstart_rust:
    la sp, _stack_top
    call rust_main
.Lhalt:
    wfi
    j .Lhalt

    .section .text.ert_workload, "ax"
    .globl __ert_workload_entry
__ert_workload_entry:
    call sha256_compress
    mv a1, a0
    li a0, -1
    ecall
"#
);

unsafe extern "C" {
    static __ert_workload_entry: u8;
}

const INPUT: [u32; 16] = [0x6162_6380, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24];

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    finish_failure(0xfffe)
}

#[unsafe(no_mangle)]
extern "C" fn rust_main() -> ! {
    if !boolar_storage_probe() {
        finish_failure(5);
    }
    let expected = sha256_compress(
        INPUT[0], INPUT[1], INPUT[2], INPUT[3], INPUT[4], INPUT[5], INPUT[6], INPUT[7], INPUT[8],
        INPUT[9], INPUT[10], INPUT[11], INPUT[12], INPUT[13], INPUT[14], INPUT[15],
    );
    let mut handler = RvDefaultHandler {
        inner: DefaultHandler {
            context: (),
            hash: no_hash,
        },
    };
    let mut regs = [[false; 32]; 32];
    let mut constants = [None; 32];
    let mut rstack = [0; 256];
    let mut vstack = [false; 65_536];
    let args = INPUT.map(|input| (word(input), None));
    let entry = ptr::addr_of!(__ert_workload_entry) as usize as u32;
    // SAFETY: QEMU maps this image at its native RV32 addresses. The workload
    // only fetches code and concrete static data from that mapped image.
    let memory = unsafe { RawMemory::new(ptr::null(), None) };

    let results = match ert_func::<_, _, 16, 2>(
        &mut handler,
        memory,
        &mut rstack,
        &mut vstack,
        entry,
        &mut regs,
        &mut constants,
        false,
        true,
        args,
    ) {
        Ok(results) => results,
        Err(_) => finish_failure(1),
    };

    if results[0].1 != Some(u32::MAX) {
        finish_failure(2);
    }
    if results[1].0 != word(expected) {
        finish_failure(3);
    }
    if results[1].1.is_some() {
        finish_failure(4);
    }
    finish_success()
}

fn word(value: u32) -> [bool; 32] {
    array::from_fn(|bit| value & (1 << bit) != 0)
}

fn no_hash(_: &mut (), _: &[[bool; 32]]) -> Result<[u8; 32], Infallible> {
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
        cells: &mut cells,
    }];
    matches!(
        execute(&mut (), &circuit, &[true, false], &mut banks),
        Ok(outputs) if outputs.as_slice() == [true] && cells == [false, false, true, false]
    )
}

fn finish_success() -> ! {
    report_heap_usage();
    finish(0x5555)
}

fn report_heap_usage() {
    uart_write(b"cirrus-ert heap_capacity_bytes=");
    uart_write_decimal(BOOLAR_HEAP_BYTES);
    uart_write(b" heap_used_bytes=");
    uart_write_decimal(ALLOCATOR.used());
    uart_write(b"\n");
}

fn uart_write(bytes: &[u8]) {
    for &byte in bytes {
        // SAFETY: QEMU's `virt` machine maps the first 16550 UART at this
        // address. Polling the line-status register keeps the metrics line
        // valid even when a runner does not drain the transmit FIFO eagerly.
        while unsafe { (0x1000_0005 as *const u8).read_volatile() } & (1 << 5) == 0 {}
        unsafe { (0x1000_0000 as *mut u8).write_volatile(byte) };
    }
}

fn uart_write_decimal(mut value: usize) {
    let mut digits = [0; 20];
    let mut count = 0;
    loop {
        digits[count] = b'0' + (value % 10) as u8;
        count += 1;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    while count > 0 {
        count -= 1;
        uart_write(&digits[count..count + 1]);
    }
}

fn finish_failure(code: u16) -> ! {
    finish((u32::from(code.max(1)) << 16) | 0x3333)
}

fn finish(status: u32) -> ! {
    // SAFETY: QEMU's `virt` machine maps the SiFive test finisher here.
    unsafe { (0x0010_0000 as *mut u32).write_volatile(status) };
    loop {
        // SAFETY: waiting is only reached if QEMU did not consume the finisher write.
        unsafe { core::arch::asm!("wfi", options(nomem, nostack)) };
    }
}
