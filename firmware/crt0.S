# Startup: set the stack, clear .bss, run main, report its return value to the simulator.
  .section .text.init
  .globl _start
_start:
  la   sp, __stack_top
  la   t0, __bss_start
  la   t1, __bss_end
1:
  bgeu t0, t1, 2f
  sw   zero, 0(t0)
  addi t0, t0, 4
  j    1b
2:
  call main
  call htif_exit
3:
  j    3b
