// Spike extension: the simd_mac custom-0 instructions, so firmware that uses them can run on
// the Spike instruction-set simulator. Semantics match hardware/rtl/simd_mac_wrapper.v:
//
//   funct7 0x00  simd_mac      rd  = rd + dot4(rs1, rs2)
//   funct7 0x01  simd_mac_clr  rd  = dot4(rs1, rs2)
//   funct7 0x02  simd_macc     acc = acc + dot4(rs1, rs2); rd = acc
//   funct7 0x03  simd_macz     acc = dot4(rs1, rs2);       rd = acc
//
// dot4 multiplies the four signed bytes of rs1 and rs2 lane by lane and adds the products.
// In the hardware, every MAC instruction leaves its result in the unit's result register, which
// is the accumulator for simd_macc; this model keeps the same single register.
// Built and loaded by firmware/Makefile:  spike --extlib=libsimd_mac.so --extension=simd_mac
#include "extension.h"

#include <cstdint>

// The Spike v1.1.0 binary does not export extension_t's destructor or its type information,
// so a library loaded with --extlib cannot link against them. Defining the class's key function
// (its out-of-line destructor) here makes the compiler emit both in this library.
extension_t::~extension_t() {}

namespace {

int32_t dot4(uint32_t a, uint32_t b) {
    int32_t sum = 0;
    for (int i = 0; i < 4; i++) sum += int32_t(int8_t(a >> (8 * i))) * int32_t(int8_t(b >> (8 * i)));
    return sum;
}

class simd_mac_t : public extension_t {
 public:
    const char *name() override { return "simd_mac"; }
    void reset() override { result = 0; }
    std::vector<insn_desc_t> get_instructions() override;
    std::vector<disasm_insn_t *> get_disasms() override;
    uint32_t result = 0;  // the unit's result register (32 bits, as in the RTL)
    simd_mac_t();
};

// One unit per simulated core; Spike runs a single hart here (-p1, the default).
simd_mac_t *g_unit = nullptr;

template <int FUNCT7>
reg_t execute(processor_t *p, insn_t insn, reg_t pc) {
    state_t *s = p->get_state();
    const uint32_t a = uint32_t(s->XPR[insn.rs1()]);
    const uint32_t b = uint32_t(s->XPR[insn.rs2()]);
    simd_mac_t *u = g_unit;
    uint32_t acc;
    switch (FUNCT7) {
        case 0: acc = uint32_t(s->XPR[insn.rd()]); break;  // simd_mac: total from rd
        case 2: acc = u->result; break;                    // simd_macc: total kept in the unit
        default: acc = 0; break;                           // simd_mac_clr, simd_macz: start fresh
    }
    u->result = acc + uint32_t(dot4(a, b));  // wraps at 32 bits, like the RTL adder
    s->XPR.write(insn.rd(), sext32(u->result));
    return pc + 4;
}

constexpr insn_bits_t MASK = 0xFE00707F;  // funct7, funct3 and opcode
constexpr insn_bits_t match(uint32_t funct7) { return (insn_bits_t(funct7) << 25) | 0x0B; }

std::vector<insn_desc_t> simd_mac_t::get_instructions() {
    return {
        {match(0), MASK, execute<0>, execute<0>},
        {match(1), MASK, execute<1>, execute<1>},
        {match(2), MASK, execute<2>, execute<2>},
        {match(3), MASK, execute<3>, execute<3>},
    };
}

// Disassembly is optional (only used by spike -l traces); the operand formatters are not
// exported by Spike's headers in this release, so none are registered.
std::vector<disasm_insn_t *> simd_mac_t::get_disasms() { return {}; }

}  // namespace

simd_mac_t::simd_mac_t() { g_unit = this; }

REGISTER_EXTENSION(simd_mac, []() { return new simd_mac_t; })
