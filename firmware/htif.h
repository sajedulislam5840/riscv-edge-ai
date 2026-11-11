// Minimal I/O through Spike's host-target interface (HTIF): print text and exit.
#ifndef HTIF_H
#define HTIF_H
#include <stdint.h>

void htif_puts(const char *s);
void htif_put_u32(uint32_t v);
void htif_put_i32(int32_t v);
void htif_exit(int code);

static inline uint32_t rdcycle(void) { uint32_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }
static inline uint32_t rdinstret(void) { uint32_t c; __asm__ volatile("rdinstret %0" : "=r"(c)); return c; }

#endif
