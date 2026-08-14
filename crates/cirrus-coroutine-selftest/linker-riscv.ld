OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

SECTIONS
{
  . = ORIGIN(RAM);

  .text.init :
  {
    KEEP(*(.text.init))
  } > RAM

  .text : ALIGN(4)
  {
    *(.text .text.*)
  } > RAM

  .rodata : ALIGN(4) { *(.rodata .rodata.*) } > RAM
  .data : ALIGN(8) { *(.data .data.*) } > RAM

  .bss (NOLOAD) : ALIGN(16)
  {
    __bss_start = .;
    *(.bss .bss.*)
    *(COMMON)
    __bss_end = .;
  } > RAM

  .stack (NOLOAD) : ALIGN(16)
  {
    . += 0x10000;
    _stack_top = .;
  } > RAM
}
