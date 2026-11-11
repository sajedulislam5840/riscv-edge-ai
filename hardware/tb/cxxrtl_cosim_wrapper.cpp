// RTL co-simulation of hardware/rtl/simd_mac_wrapper.v (the custom-0 decoder + MAC unit).
//
// Replays every golden vector as a real instruction stream:
//   pass 1: simd_macz / simd_macc (internal accumulator). The rd input carries random garbage
//           to prove these forms never read it, and idle cycles and unrelated instructions are
//           mixed in at random to prove the running total is kept between MAC instructions.
//   pass 2: the original simd_mac_clr / simd_mac forms, with rd supplying the accumulator.
// Build and run: hardware/scripts/cosim.sh
#include "simd_mac_wrapper_cxxrtl.cc"
#include <cstdint>
#include <cstdio>
#include <vector>

struct Vec { uint32_t rs1, rs2, acc_in, clr, expected; };

static uint32_t rng_state = 12345;
static uint32_t rnd() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; return rng_state; }

static uint32_t encode(uint32_t funct7, uint32_t rd, uint32_t rs1, uint32_t rs2) {
    return funct7 << 25 | rs2 << 20 | rs1 << 15 | 0u << 12 | rd << 7 | 0x0Bu;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "hardware/tb/golden_vectors.hex";
    FILE *f = std::fopen(path, "r");
    if (!f) { std::printf("cannot open %s\n", path); return 2; }
    std::vector<Vec> v;
    char line[256];
    while (std::fgets(line, sizeof line, f)) {
        Vec x;
        if (std::sscanf(line, "%x %x %x %u %x", &x.rs1, &x.rs2, &x.acc_in, &x.clr, &x.expected) == 5) v.push_back(x);
    }
    std::fclose(f);

    cxxrtl_design::p_simd__mac__wrapper dut;
    // The extra step() lets output ports that are computed from registers settle after the edge.
    auto tick = [&] { dut.p_clk.set<bool>(false); dut.step(); dut.p_clk.set<bool>(true); dut.step(); dut.step(); };
    dut.p_rst__n.set<bool>(false); dut.p_req__valid.set<bool>(false); tick();
    dut.p_rst__n.set<bool>(true);

    long errors = 0, idle = 0;
    auto issue = [&](uint32_t insn, uint32_t rs1, uint32_t rs2, uint32_t rd_val, uint32_t expected, long n) {
        dut.p_insn.set<uint32_t>(insn);
        dut.p_rs1__val.set<uint32_t>(rs1);
        dut.p_rs2__val.set<uint32_t>(rs2);
        dut.p_rd__val.set<uint32_t>(rd_val);
        dut.p_req__valid.set<bool>(true);
        tick();
        uint32_t got = dut.p_rsp__data.get<uint32_t>();
        if (!dut.p_rsp__valid.get<bool>() || got != expected) {
            if (errors < 10) std::printf("FAIL vector %ld: got %08x expected %08x\n", n, got, expected);
            ++errors;
        }
    };

    // Pass 1: internal accumulator.
    for (size_t i = 0; i < v.size(); ++i) {
        if (rnd() % 8 == 0) {  // an idle cycle or an unrelated instruction (an ADD) in between
            dut.p_insn.set<uint32_t>(0x00b50533u);
            dut.p_rd__val.set<uint32_t>(rnd());
            dut.p_req__valid.set<bool>(rnd() & 1);
            tick();
            ++idle;
        }
        const uint32_t funct7 = v[i].clr ? 0x03 : 0x02;  // simd_macz : simd_macc
        issue(encode(funct7, 10, 11, 12), v[i].rs1, v[i].rs2, rnd(), v[i].expected, long(i + 1));
    }
    const long pass1_errors = errors;

    // Pass 2: original forms, rd carries the running total.
    for (size_t i = 0; i < v.size(); ++i) {
        const uint32_t funct7 = v[i].clr ? 0x01 : 0x00;  // simd_mac_clr : simd_mac
        issue(encode(funct7, 10, 11, 12), v[i].rs1, v[i].rs2, v[i].clr ? rnd() : v[i].acc_in, v[i].expected,
              long(v.size() + i + 1));
    }

    std::printf("Wrapper, internal accumulator (simd_macz/simd_macc): %zu instructions, %ld idle or unrelated cycles, mismatches: %ld\n",
                v.size(), idle, pass1_errors);
    std::printf("Wrapper, rd accumulator (simd_mac_clr/simd_mac):     %zu instructions, mismatches: %ld\n",
                v.size(), errors - pass1_errors);
    return errors == 0 && !v.empty() ? 0 : 1;
}
