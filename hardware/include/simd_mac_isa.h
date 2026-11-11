// =============================================================================
// File:    simd_mac_isa.h
// Brief:   RISC-V Custom-0 SIMD MAC ISA Extension — Inline Assembly Macros
// Target:  RV32I + custom-0 co-processor (Opcode 0x0B, funct3=0)
//
// Instruction Set:
//   simd_mac      rd, rs1, rs2  |  funct7=0x00  |  rd += sum(rs1[i]*rs2[i])
//   simd_mac_clr  rd, rs1, rs2  |  funct7=0x01  |  rd  = sum(rs1[i]*rs2[i])
//   simd_macc     rd, rs1, rs2  |  funct7=0x02  |  acc += sum(rs1[i]*rs2[i]); rd = acc
//   simd_macz     rd, rs1, rs2  |  funct7=0x03  |  acc  = sum(rs1[i]*rs2[i]); rd = acc
//
// simd_macc / simd_macz keep the running total inside the unit, so they read only rs1 and rs2
// (two register read ports, as on PicoRV32's PCPI). simd_mac reads rd as a third operand.
//
// Register Packing Convention (little-endian):
//   Byte 0 [7:0]  : INT8 lane 0
//   Byte 1 [15:8] : INT8 lane 1
//   Byte 2 [23:16]: INT8 lane 2
//   Byte 3 [31:24]: INT8 lane 3
//
// Usage:
//   #include "simd_mac_isa.h"
//
//   // Start fresh dot product:
//   int32_t result = 0;
//   SIMD_MAC_CLR(result, a_packed, b_packed);
//
//   // Accumulate across loop iterations:
//   for (int i = 0; i < n/4; i++)
//       SIMD_MAC(result, a[i], b[i]);
// =============================================================================

#ifndef SIMD_MAC_ISA_H
#define SIMD_MAC_ISA_H

#include <stdint.h>

// Custom-0 opcode constant for assembler
#define CUSTOM_0 0x0B
// Two-step stringify so the assembler sees "0x0B", not the macro name.
#define SIMD_MAC_STR_(x) #x
#define SIMD_MAC_STR(x) SIMD_MAC_STR_(x)

// Packing helpers, used by both the hardware and the software paths.
// Pack four signed int8_t values into one uint32_t (little-endian)
static inline uint32_t simd_pack_i8(int8_t b0, int8_t b1, int8_t b2, int8_t b3) {
    return ((uint32_t)(uint8_t)b0)
         | ((uint32_t)(uint8_t)b1 << 8)
         | ((uint32_t)(uint8_t)b2 << 16)
         | ((uint32_t)(uint8_t)b3 << 24);
}

// Signed 8-bit extract from packed word at byte lane [lane]
static inline int32_t simd_extract_i8(uint32_t packed, int lane) {
    return (int32_t)(int8_t)((packed >> (lane * 8)) & 0xFF);
}


// =============================================================================
// Hardware Path: GCC inline assembly for RISC-V targets
// Use when compiling with riscv32-unknown-elf-gcc targeting this co-processor
// =============================================================================
#if defined(__riscv) && defined(SIMD_MAC_HW_ENABLE)

// simd_mac: rd += sum(rs1[i] * rs2[i]) for i in [0..3]
// Encoding: funct7=0x00, funct3=0, opcode=custom-0
#define SIMD_MAC(rd, rs1, rs2) \
    __asm__ volatile ( \
        ".insn r " SIMD_MAC_STR(CUSTOM_0) ", 0, 0, %0, %1, %2" \
        : "+r"(rd) \
        : "r"(rs1), "r"(rs2) \
    )

// simd_mac_clr: rd = sum(rs1[i] * rs2[i]) for i in [0..3]  (accumulator cleared)
// Encoding: funct7=0x01, funct3=0, opcode=custom-0
#define SIMD_MAC_CLR(rd, rs1, rs2) \
    __asm__ volatile ( \
        ".insn r " SIMD_MAC_STR(CUSTOM_0) ", 0, 1, %0, %1, %2" \
        : "=r"(rd) \
        : "r"(rs1), "r"(rs2) \
    )

// simd_macz: start a new running total inside the unit; rd = that total
// Encoding: funct7=0x03, funct3=0, opcode=custom-0
#define SIMD_MACZ(rd, rs1, rs2) \
    __asm__ volatile ( \
        ".insn r " SIMD_MAC_STR(CUSTOM_0) ", 0, 3, %0, %1, %2" \
        : "=r"(rd) \
        : "r"(rs1), "r"(rs2) \
    )

// simd_macc: add to the running total inside the unit; rd = the new total
// Encoding: funct7=0x02, funct3=0, opcode=custom-0
#define SIMD_MACC(rd, rs1, rs2) \
    __asm__ volatile ( \
        ".insn r " SIMD_MAC_STR(CUSTOM_0) ", 0, 2, %0, %1, %2" \
        : "=r"(rd) \
        : "r"(rs1), "r"(rs2) \
    )

// =============================================================================
// Software Fallback Path: Bit-exact C implementation for host compilation
// Use for unit testing, golden reference, and cross-compilation compatibility
// =============================================================================
#else

// simd_mac_sw: accumulate dot product (software fallback)
static inline int32_t simd_mac_sw(int32_t acc, uint32_t rs1, uint32_t rs2) {
    int32_t sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += simd_extract_i8(rs1, i) * simd_extract_i8(rs2, i);
    }
    return acc + sum;
}

// simd_mac_clr_sw: clear and start fresh dot product (software fallback)
static inline int32_t simd_mac_clr_sw(uint32_t rs1, uint32_t rs2) {
    int32_t sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += simd_extract_i8(rs1, i) * simd_extract_i8(rs2, i);
    }
    return sum;
}

// Model of the unit's internal running total, used by simd_macz / simd_macc.
static int32_t simd_internal_acc_sw = 0;

static inline int32_t simd_macz_sw(uint32_t rs1, uint32_t rs2) {
    simd_internal_acc_sw = simd_mac_clr_sw(rs1, rs2);
    return simd_internal_acc_sw;
}

static inline int32_t simd_macc_sw(uint32_t rs1, uint32_t rs2) {
    simd_internal_acc_sw = simd_mac_sw(simd_internal_acc_sw, rs1, rs2);
    return simd_internal_acc_sw;
}

// Map hardware macros to software fallbacks transparently
#define SIMD_MAC(rd, rs1, rs2)     do { (rd) = simd_mac_sw((rd), (rs1), (rs2)); } while(0)
#define SIMD_MAC_CLR(rd, rs1, rs2) do { (rd) = simd_mac_clr_sw((rs1), (rs2)); } while(0)
#define SIMD_MACZ(rd, rs1, rs2)    do { (rd) = simd_macz_sw((rs1), (rs2)); } while(0)
#define SIMD_MACC(rd, rs1, rs2)    do { (rd) = simd_macc_sw((rs1), (rs2)); } while(0)

#endif  // __riscv && SIMD_MAC_HW_ENABLE

// =============================================================================
// Utility: Dot product of two int8 arrays of length n (must be multiple of 4)
// Uses SIMD_MAC / SIMD_MAC_CLR for HW acceleration or falls back to software
// =============================================================================
static inline int32_t simd_dot4(const int8_t* a, const int8_t* b, int n) {
    int32_t acc = 0;
    int i = 0;
    // Process 4 elements at a time using packed MAC
    for (; i <= n - 4; i += 4) {
        uint32_t pa = simd_pack_i8(a[i], a[i+1], a[i+2], a[i+3]);
        uint32_t pb = simd_pack_i8(b[i], b[i+1], b[i+2], b[i+3]);
        if (i == 0) {
            SIMD_MAC_CLR(acc, pa, pb);
        } else {
            SIMD_MAC(acc, pa, pb);
        }
    }
    // Scalar tail for remainder (n not multiple of 4)
    for (; i < n; i++) {
        acc += (int32_t)a[i] * (int32_t)b[i];
    }
    return acc;
}

#endif  // SIMD_MAC_ISA_H
