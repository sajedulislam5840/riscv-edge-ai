"""Cycle trace of one neuron on the PicoRV32 + MAC SoC, for the website.

  python scripts/make_trace.py --find-pcs firmware/build/pico_mnist.elf
      print the first instruction of the neuron loop and the address where the dot product ends
  python scripts/make_trace.py build/trace_scalar.tsv build/trace_mac.tsv > web/js/trace.js
      turn the two recorded windows (from hardware/scripts/soc_trace.sh) into web/js/trace.js
"""
import json
import re
import subprocess
import sys

LOADS = {0: "lb", 1: "lh", 2: "lw", 4: "lbu", 5: "lhu"}
STORES = {0: "sb", 1: "sh", 2: "sw"}
BRANCH = {0: "beq", 1: "bne", 4: "blt", 5: "bge", 6: "bltu", 7: "bgeu"}
CUSTOM = {1: "simd_mac_clr", 2: "simd_macc", 3: "simd_macz", 0: "simd_mac"}
ALU_I = {0: "addi", 1: "slli", 2: "slti", 3: "sltiu", 4: "xori", 6: "ori", 7: "andi"}
ALU_R = {(0, 0): "add", (0x20, 0): "sub", (0, 1): "sll", (0, 2): "slt", (0, 3): "sltu", (0, 4): "xor",
         (0, 5): "srl", (0x20, 5): "sra", (0, 6): "or", (0, 7): "and"}
MUL = {0: "mul", 1: "mulh", 2: "mulhsu", 3: "mulhu"}


def decode(w):
    """(mnemonic, class) of one instruction word. Classes: L load, S store, M multiply, X custom MAC, B branch or jump, A other."""
    op, f3, f7 = w & 0x7F, (w >> 12) & 7, w >> 25
    if op == 0x03:
        return LOADS.get(f3, "load"), "L"
    if op == 0x23:
        return STORES.get(f3, "store"), "S"
    if op == 0x63:
        return BRANCH.get(f3, "branch"), "B"
    if op in (0x6F, 0x67):
        return ("jal" if op == 0x6F else "jalr"), "B"
    if op == 0x0B:
        return CUSTOM.get(f7, "custom"), "X"
    if op == 0x33:
        if f7 == 1 and f3 in MUL:
            return MUL[f3], "M"
        if f7 == 1:
            return ("div", "divu", "rem", "remu")[f3 - 4], "M"
        return ALU_R.get((f7, f3), "alu"), "A"
    if op == 0x13:
        return ("srai" if f3 == 5 and f7 == 0x20 else "srli" if f3 == 5 else ALU_I.get(f3, "alui")), "A"
    if op == 0x37:
        return "lui", "A"
    if op == 0x17:
        return "auipc", "A"
    return "other", "A"


def find_pcs(elf):
    dis = subprocess.run(["riscv-none-elf-objdump", "-d", "--no-show-raw-insn", elf], capture_output=True, text=True, check=True).stdout
    body, inside = [], False
    for line in dis.splitlines():
        if re.match(r"[0-9a-f]+ <infer_opt>:", line):
            inside = True
            continue
        if inside and re.match(r"[0-9a-f]+ <", line):
            break
        m = re.match(r"\s+([0-9a-f]+):\s+(\S+)\s*(.*)", line) if inside else None
        if m:
            body.append((int(m.group(1), 16), m.group(2), m.group(3)))
    target = lambda ops: int(re.search(r"([0-9a-f]+) <", ops).group(1), 16)
    index = {a: i for i, (a, _, _) in enumerate(body)}
    blocks = sorted({target(o) for _, m, o in body if m == "beq" and target(o) in index
                     and any(b[1] == ".insn" for b in body[index[target(o)]:index[target(o)] + 4])})
    start, block = min((next(x for x, m, o in body if m == "beq" and target(o) == blk), blk) for blk in blocks)
    end = next(target(o) for x, m, o in body if m == "j" and x > block)
    return start, end


def window(path):
    fetches, data = [], []
    for line in open(path):
        kind, cyc, a, b = line.split()
        (fetches if kind == "I" else data).append((int(cyc), int(a), int(b)))
    return fetches, data


def build(path, names):
    fetches, data = window(path)
    pcs, ops, cycles = [], [], []
    for i in range(len(fetches) - 1):
        cyc, pc, word = fetches[i]
        mn, cls = decode(word)
        if mn not in names:
            names.append(mn)
        pcs.append(pc)
        ops.append((names.index(mn), cls))
        cycles.append(fetches[i + 1][0] - cyc)
    return {"pc": pcs, "op": [o[0] for o in ops], "cls": "".join(o[1] for o in ops), "c": cycles,
            "total": fetches[-1][0] - fetches[0][0], "dataAccesses": len(data)}


def main():
    if sys.argv[1] == "--find-pcs":
        s, e = find_pcs(sys.argv[2])
        print(hex(s), hex(e))
        return
    names = []
    out = {
        "meta": {"core": "PicoRV32 with the MAC unit, RTL simulation, memory with no wait states",
                 "image": 0, "layer": 1, "neuron": 0, "macs": 784},
        "scalar": build(sys.argv[1], names),
        "mac": build(sys.argv[2], names),
    }
    out["names"] = names
    print("// Cycle trace of layer 1, neuron 0, test image 0, recorded from the RTL simulation by hardware/scripts/soc_trace.sh.")
    print("window.CYCLE_TRACE = " + json.dumps(out, separators=(",", ":")) + ";")


if __name__ == "__main__":
    main()
