ENTRY(reset_handler)

MEMORY
{
  FLASH (rx) : ORIGIN = 0x10000000, LENGTH = 4M
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

  . = ORIGIN(RAM);
  .text : ALIGN(4) { *(.text .text.*) } > RAM
  .rodata : ALIGN(4) { *(.rodata .rodata.*) } > RAM
  .data : ALIGN(4) { *(.data .data.*) } > RAM

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
