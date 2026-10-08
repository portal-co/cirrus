OUTPUT_ARCH(aarch64)
ENTRY(_start)

MEMORY
{
    ram (rwx) : ORIGIN = 0x40000000, LENGTH = 128M
}

SECTIONS
{
    .text : {
        KEEP(*(.text.init))
        *(.text .text.*)
    } > ram
    .rodata : { *(.rodata .rodata.*) } > ram
    .data : { *(.data .data.*) } > ram
    .bss : {
        __bss_start = .;
        *(.bss .bss.* COMMON)
        __bss_end = .;
    } > ram
    . = ALIGN(16);
    . += 64K;
    _stack_top = .;
}
