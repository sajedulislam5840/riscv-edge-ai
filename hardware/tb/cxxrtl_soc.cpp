// Whole-SoC simulation: PicoRV32 + SIMD MAC (hardware/soc/pico_mac_soc.v) running a program.
//
// The testbench provides 1 MiB of RAM at address 0 and two memory-mapped registers:
//   0x10000000  write a byte: print it          0x10000004  write: stop with this exit code
// Memory answers like PicoRV32's own testbench: ready one cycle after valid (0 extra wait
// states); an optional third argument adds N wait states to every access.
//
// Build and run: hardware/scripts/soc_sim.sh
//   usage: cxxrtl_soc program.bin [max_cycles] [wait_states]
#include "soc_cxxrtl.cc"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const uint32_t RAM_BYTES = 1u << 20;
static const uint32_t UART_TX = 0x10000000u, EXIT_REG = 0x10000004u;

int main(int argc, char **argv) {
    if (argc < 2) { std::printf("usage: %s program.bin [max_cycles] [wait_states]\n", argv[0]); return 2; }
    const uint64_t max_cycles = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 2000000000ull;
    const unsigned wait_states = argc > 3 ? (unsigned)std::strtoul(argv[3], nullptr, 10) : 0;

    std::vector<uint8_t> ram(RAM_BYTES, 0);
    FILE *f = std::fopen(argv[1], "rb");
    if (!f) { std::printf("cannot open %s\n", argv[1]); return 2; }
    size_t n = std::fread(ram.data(), 1, ram.size(), f);
    std::fclose(f);
    std::printf("[soc] loaded %zu bytes, %u extra wait state(s) per memory access\n", n, wait_states);

    cxxrtl_design::p_pico__mac__soc dut;
    auto settle = [&] { dut.step(); };
    dut.p_clk.set<bool>(false);
    dut.p_resetn.set<bool>(false);
    dut.p_mem__ready.set<bool>(false);
    dut.p_mem__rdata.set<uint32_t>(0);
    settle();
    for (int i = 0; i < 10; i++) {
        dut.p_clk.set<bool>(true); settle();
        dut.p_clk.set<bool>(false); settle();
    }
    dut.p_resetn.set<bool>(true);
    settle();

    bool ready = false, finished = false;
    unsigned waited = 0;
    int exit_code = -1;
    uint64_t cycles = 0;

    while (!finished && cycles < max_cycles) {
        // What the memory does at this clock edge, from the values just before it.
        bool next_ready = false;
        uint32_t next_rdata = 0;
        if (dut.p_mem__valid.get<bool>() && !ready) {
            if (waited >= wait_states) {
                waited = 0;
                next_ready = true;
                uint32_t addr = dut.p_mem__addr.get<uint32_t>();
                uint32_t wstrb = dut.p_mem__wstrb.get<uint32_t>();
                uint32_t wdata = dut.p_mem__wdata.get<uint32_t>();
                if (wstrb == 0) {
                    if (addr < RAM_BYTES - 3)
                        next_rdata = ram[addr] | ram[addr + 1] << 8 | ram[addr + 2] << 16 | (uint32_t)ram[addr + 3] << 24;
                } else if (addr == UART_TX) {
                    std::putchar((int)(wdata & 0xFF));
                } else if (addr == EXIT_REG) {
                    exit_code = (int)wdata; finished = true;
                } else if (addr < RAM_BYTES - 3) {
                    for (int b = 0; b < 4; b++) if (wstrb >> b & 1) ram[addr + b] = (uint8_t)(wdata >> (8 * b));
                } else {
                    std::printf("\n[soc] write outside RAM at 0x%08x\n", addr); exit_code = 3; finished = true;
                }
            } else {
                waited++;
            }
        }
        dut.p_clk.set<bool>(true);  settle();
        ready = next_ready;
        dut.p_mem__ready.set<bool>(ready);
        dut.p_mem__rdata.set<uint32_t>(next_rdata);
        settle();
        dut.p_clk.set<bool>(false); settle();
        cycles++;
        if (dut.p_trap.get<bool>()) {
            std::printf("\n[soc] TRAP after %llu cycles (illegal instruction or misaligned access)\n", (unsigned long long)cycles);
            return 4;
        }
    }
    std::fflush(stdout);
    if (!finished) { std::printf("\n[soc] timeout after %llu cycles\n", (unsigned long long)cycles); return 5; }
    std::printf("[soc] program exited with code %d after %llu clock cycles\n", exit_code, (unsigned long long)cycles);
    return exit_code == 0 ? 0 : 1;
}
