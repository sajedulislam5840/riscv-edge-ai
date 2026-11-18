#include "soc_cxxrtl.cc"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static const uint32_t RAM_BYTES = 1u << 20;
static const uint32_t UART_TX = 0x10000000u, EXIT_REG = 0x10000004u;

int main(int argc, char **argv) {
    std::vector<const char *> pos;
    uint32_t opt_examples = 0xFFFFFFFFu, opt_scalar_only = 0xFFFFFFFFu;
    const char *trace_path = nullptr;
    uint32_t trace_start = 0, trace_end = 0, trace_skip = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--examples" && i + 1 < argc) opt_examples = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (a == "--scalar-only") opt_scalar_only = 1;
        else if (a == "--trace" && i + 1 < argc) trace_path = argv[++i];
        else if (a == "--trace-start" && i + 1 < argc) trace_start = (uint32_t)std::strtoul(argv[++i], nullptr, 0);
        else if (a == "--trace-end" && i + 1 < argc) trace_end = (uint32_t)std::strtoul(argv[++i], nullptr, 0);
        else if (a == "--trace-skip" && i + 1 < argc) trace_skip = (uint32_t)std::strtoul(argv[++i], nullptr, 0);
        else pos.push_back(argv[i]);
    }
    if (pos.empty()) { std::printf("usage: %s program.bin [max_cycles] [wait_states] [--examples N] [--scalar-only]\n       [--trace file --trace-start PC --trace-end PC [--trace-skip N]]\n", argv[0]); return 2; }
    const uint64_t max_cycles = pos.size() > 1 ? std::strtoull(pos[1], nullptr, 10) : 2000000000ull;
    const unsigned wait_states = pos.size() > 2 ? (unsigned)std::strtoul(pos[2], nullptr, 10) : 0;

    std::vector<uint8_t> ram(RAM_BYTES, 0);
    FILE *f = std::fopen(pos[0], "rb");
    if (!f) { std::printf("cannot open %s\n", pos[0]); return 2; }
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
    FILE *trace = trace_path ? std::fopen(trace_path, "w") : nullptr;
    bool tracing = false, trace_done = false;
    uint32_t start_seen = 0, last_pc = 0xFFFFFFFFu;

    while (!finished && cycles < max_cycles) {
        bool next_ready = false;
        uint32_t next_rdata = 0;
        if (dut.p_mem__valid.get<bool>() && !ready) {
            if (waited >= wait_states) {
                waited = 0;
                next_ready = true;
                uint32_t addr = dut.p_mem__addr.get<uint32_t>();
                uint32_t wstrb = dut.p_mem__wstrb.get<uint32_t>();
                uint32_t wdata = dut.p_mem__wdata.get<uint32_t>();
                if (trace && tracing && !dut.p_mem__instr.get<bool>())
                    std::fprintf(trace, "D\t%llu\t%u\t%u\n", (unsigned long long)cycles, addr, wstrb ? 1u : 0u);
                if (wstrb == 0) {
                    if (addr == 0x10000008u) next_rdata = opt_examples;
                    else if (addr == 0x1000000Cu) next_rdata = opt_scalar_only;
                    else if (addr < RAM_BYTES - 3)
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
        if (trace && !trace_done && dut.p_cpu_2e_reg__pc.get<uint32_t>() != last_pc) {
            const uint32_t pc = dut.p_cpu_2e_reg__pc.get<uint32_t>();
            last_pc = pc;
            if (!tracing && pc == trace_start && start_seen++ == trace_skip) tracing = true;
            if (tracing) {
                const uint32_t word = pc < RAM_BYTES - 3 ? (ram[pc] | ram[pc + 1] << 8 | ram[pc + 2] << 16 | (uint32_t)ram[pc + 3] << 24) : 0;
                std::fprintf(trace, "I\t%llu\t%u\t%u\n", (unsigned long long)cycles, pc, word);
                if (pc == trace_end) { tracing = false; trace_done = true; finished = true; exit_code = 0; }
            }
        }
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
    if (trace) std::fclose(trace);
    if (!finished) { std::printf("\n[soc] timeout after %llu cycles\n", (unsigned long long)cycles); return 5; }
    std::printf("[soc] program exited with code %d after %llu clock cycles\n", exit_code, (unsigned long long)cycles);
    return exit_code == 0 ? 0 : 1;
}
