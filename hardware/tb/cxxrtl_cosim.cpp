// RTL co-simulation of hardware/rtl/simd_mac.v using Yosys CXXRTL.
// Streams every vector in golden_vectors.hex through the compiled RTL and
// compares mac_out against the C++ golden result. Build: hardware/scripts/cosim.sh
#include "simd_mac_cxxrtl.cc"
#include <cstdio>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "hardware/tb/golden_vectors.hex";
    FILE *f = std::fopen(path, "r");
    if (!f) { std::printf("cannot open %s\n", path); return 2; }

    cxxrtl_design::p_simd__mac dut;
    auto tick = [&] {
        dut.p_clk.set<bool>(false); dut.step();
        dut.p_clk.set<bool>(true);  dut.step();
    };

    dut.p_rst__n.set<bool>(false); tick();
    dut.p_rst__n.set<bool>(true);

    char line[256];
    long n = 0, errors = 0;
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        unsigned rs1, rs2, acc, clr, expected;
        if (std::sscanf(line, "%x %x %x %u %x", &rs1, &rs2, &acc, &clr, &expected) != 5) continue;
        dut.p_rs1__data.set<uint32_t>(rs1);
        dut.p_rs2__data.set<uint32_t>(rs2);
        dut.p_acc__in.set<uint32_t>(acc);
        dut.p_clr__acc.set<bool>(clr);
        dut.p_enable.set<bool>(true);
        tick();
        uint32_t got = dut.p_mac__out.get<uint32_t>();
        ++n;
        if (got != expected) {
            if (errors < 10) std::printf("FAIL vector %ld: got %08x expected %08x\n", n, got, expected);
            ++errors;
        }
    }
    std::fclose(f);
    std::printf("RTL vectors: %ld  mismatches: %ld\n", n, errors);
    return errors == 0 && n > 0 ? 0 : 1;
}
