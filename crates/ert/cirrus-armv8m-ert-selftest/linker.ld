ENTRY(reset_handler)

MEMORY
{
  FLASH (rx) : ORIGIN = 0x10000000, LENGTH = 4M
  WORKLOAD (rwx) : ORIGIN = 0x20000000, LENGTH = 32K
  RAM (rwx) : ORIGIN = 0x28000000, LENGTH = 2M
}

SECTIONS
{
  .vector_table ORIGIN(FLASH) : ALIGN(256)
  {
    KEEP(*(.vector_table))
  } > FLASH

  .text.init :
  {
    KEEP(*(.text.init))
  } > FLASH

  . = ORIGIN(WORKLOAD);
  .text.ert_workload :
  {
    KEEP(*(.text.ert_workload))
    *(.text.sha256_compress)
    *(.text._ZN25cirrus_ert_sha256_fixture*)
  } > WORKLOAD

  .rodata.ert_workload : ALIGN(4)
  {
    *(.rodata._ZN25cirrus_ert_sha256_fixture*)
  } > WORKLOAD

  . = ORIGIN(RAM);
  .text :
  {
    *(.text .text.*)
  } > RAM

  .rodata : ALIGN(4) { *(.rodata .rodata.*) } > RAM
  .data : ALIGN(4) { *(.data .data.*) } > RAM
  .bss (NOLOAD) : ALIGN(8)
  {
    *(.bss .bss.*)
    *(COMMON)
  } > RAM
  .stack (NOLOAD) : ALIGN(16)
  {
    . += 0x40000;
    _stack_top = .;
  } > RAM
}
