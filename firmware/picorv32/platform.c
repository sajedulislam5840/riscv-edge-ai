// I/O for the PicoRV32 SoC testbench: same functions as htif.c, but through two memory-mapped
// registers that the testbench watches. Writing a byte to UART_TX prints it; writing to EXIT_REG
// ends the simulation with that exit code.
#include "../htif.h"

#define UART_TX  (*(volatile uint32_t *)0x10000000u)
#define EXIT_REG (*(volatile uint32_t *)0x10000004u)

static void put_char(char c) { UART_TX = (uint32_t)(unsigned char)c; }

void htif_puts(const char *s) { while (*s) put_char(*s++); }

void htif_put_u32(uint32_t v) {
    char buf[12];
    int i = 11;
    buf[i] = 0;
    do { buf[--i] = (char)('0' + v % 10); v /= 10; } while (v);
    htif_puts(&buf[i]);
}

void htif_put_i32(int32_t v) {
    if (v < 0) { htif_puts("-"); htif_put_u32((uint32_t)(-(int64_t)v)); }
    else htif_put_u32((uint32_t)v);
}

void htif_exit(int code) {
    EXIT_REG = (uint32_t)code;
    for (;;) {}
}

// The compiler may emit calls to these even with -ffreestanding.
void *memset(void *d, int c, unsigned n) { unsigned char *p = d; while (n--) *p++ = (unsigned char)c; return d; }
void *memcpy(void *d, const void *s, unsigned n) { unsigned char *p = d; const unsigned char *q = s; while (n--) *p++ = *q++; return d; }
