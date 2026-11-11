#include "htif.h"

// Spike polls these two symbols. Each request is the address of a small syscall block,
// which keeps every write a single word with the top half zero.
volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));

static volatile uint64_t magic[8] __attribute__((aligned(64)));

static void htif_syscall(uint64_t n, uint64_t a0, uint64_t a1, uint64_t a2) {
    magic[0] = n; magic[1] = a0; magic[2] = a1; magic[3] = a2;
    __sync_synchronize();
    tohost = (uint32_t)(uintptr_t)magic;
    while (fromhost == 0) {}
    fromhost = 0;
    __sync_synchronize();
}

void htif_puts(const char *s) {
    uint32_t n = 0;
    while (s[n]) n++;
    htif_syscall(64 /* write */, 1, (uint32_t)(uintptr_t)s, n);
}

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
    tohost = ((uint32_t)code << 1) | 1;
    for (;;) {}
}

// The compiler may emit calls to these even with -ffreestanding.
void *memset(void *d, int c, unsigned n) { unsigned char *p = d; while (n--) *p++ = (unsigned char)c; return d; }
void *memcpy(void *d, const void *s, unsigned n) { unsigned char *p = d; const unsigned char *q = s; while (n--) *p++ = *q++; return d; }
