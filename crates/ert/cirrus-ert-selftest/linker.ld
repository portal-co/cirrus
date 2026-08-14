OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

SECTIONS
{
  . = ORIGIN(RAM);

  /* QEMU's riscv virt machine jumps straight to ORIGIN(RAM) for `-kernel`
     boot regardless of the ELF entry point, so `_start` (from `.text.init`)
     must stay the very first thing linked here. */
  .text.init :
  {
    KEEP(*(.text.init))
  } > RAM

  .text.ert_workload :
  {
    KEEP(*(.text.ert_workload))
    *(.text.sha256_compress)
    *(.text._ZN25cirrus_ert_sha256_fixture*)
  } > RAM

  .rodata.ert_workload : ALIGN(4)
  {
    *(.rodata._ZN25cirrus_ert_sha256_fixture*)
  } > RAM

  .text : ALIGN(4)
  {
    *(.text .text.*)
  } > RAM

  .rodata : ALIGN(4)
  {
    *(.rodata .rodata.*)
  } > RAM

  .data : ALIGN(4)
  {
    *(.data .data.*)
  } > RAM

  .bss (NOLOAD) : ALIGN(4)
  {
    __bss_start = .;
    *(.bss .bss.*)
    *(COMMON)
    __bss_end = .;
  } > RAM

  .stack (NOLOAD) : ALIGN(16)
  {
    . += 0x40000;
    _stack_top = .;
  } > RAM
}
