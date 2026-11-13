/* Bare-metal RV32 memory map for the PicoRV32 SoC testbench: 1 MiB of RAM at address 0. */
OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY { RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 1M }

SECTIONS
{
  .text : { *(.text.init) *(.text .text.*) } > RAM
  .rodata : { *(.rodata .rodata.*) *(.srodata .srodata.*) } > RAM
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
