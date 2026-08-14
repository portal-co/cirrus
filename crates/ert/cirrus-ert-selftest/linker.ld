OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

SECTIONS
{
  . = ORIGIN(RAM);

  .text :
  {
    KEEP(*(.text.init))
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
