/* Bare-metal RV32 memory map for Spike: everything in RAM at 0x80000000. */
OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 2M }

SECTIONS
{
  .text : { *(.text.init) *(.text .text.*) } > RAM
  .rodata : { *(.rodata .rodata.*) *(.srodata .srodata.*) } > RAM
  .tohost ALIGN(64) : { *(.tohost) } > RAM
  .data : { *(.data .data.*) *(.sdata .sdata.*) } > RAM
  .bss (NOLOAD) : ALIGN(8) {
    __bss_start = .;
    *(.bss .bss.*) *(.sbss .sbss.*) *(COMMON)
    __bss_end = .;
  } > RAM
  . = ALIGN(16);
  . = . + 64K;
  __stack_top = .;
}
