"""Record measured results for the web demo in web/js/results.js.

Parses the output of the project's own commands, so the page shows what the code produced:
  make -C firmware run                 > build/firmware_run.txt
  bash hardware/scripts/cosim.sh       > build/cosim.txt
  bash hardware/scripts/soc_sim.sh isa   > build/soc_isa.txt     (PicoRV32 SoC, RTL simulation)
  bash hardware/scripts/soc_sim.sh mnist > build/soc_mnist.txt
  bash hardware/scripts/soc_sweep.sh     > build/soc_sweep.txt    (optional)
  bash hardware/scripts/soc_synth.sh     > build/soc_synth.txt    (optional)
  bash hardware/scripts/fmax.sh        > build/fmax.txt      (optional)

  python scripts/record_results.py           write web/js/results.js
  python scripts/record_results.py --check   fail if the firmware or co-simulation numbers differ
                                             from the committed file (used by CI; checks whichever
                                             outputs exist; Fmax is left out because it depends on
                                             tool builds)
"""
import json
import os
import re
import sys

OUT = "web/js/results.js"


def read(path):
    return open(path, encoding="utf-8").read() if os.path.exists(path) else None


def firmware(text):
    lines = [l for l in text.splitlines()
             if l.startswith(("riscv-edge-ai", "test image", "one inference", "with ", "optimised", "multiply", "RESULT"))]
    images = [l for l in lines if l.startswith("test image")]
    return {
        "instructionsPerInference": int(re.search(r"one inference: (\d+) instructions", text).group(1)),
        "macClrAddInstructionsPerInference": int(re.search(r"with simd_mac_clr \+ add: (\d+) instructions", text).group(1)),
        "macInstructionsPerInference": int(re.search(r"with simd_macz/simd_macc: (\d+) instructions", text).group(1)),
        "optScalarInstructionsPerInference": int(re.search(r"optimised scalar \(unrolled\): (\d+) instructions", text).group(1)),
        "optMacInstructionsPerInference": int(re.search(r"optimised MAC \(unrolled simd_macz/simd_macc\): (\d+) instructions", text).group(1)),
        "macsPerInference": int(re.search(r"multiply-accumulates per inference: (\d+)", text).group(1)),
        "imagesChecked": len(images),
        "imagesCorrect": sum(1 for l in images if re.search(r"label (\d+), predicted \1,", l)),
        "allMatch": "RESULT: all outputs match" in text,
        "log": lines,
    }


def cosim(text):
    unit = re.search(r"RTL vectors: (\d+)\s+mismatches: (\d+)", text)
    acc = re.search(r"internal accumulator .*?:\s+(\d+) instructions, (\d+) idle or unrelated cycles, mismatches: (\d+)", text)
    rd = re.search(r"rd accumulator .*?:\s+(\d+) instructions, mismatches: (\d+)", text)
    return {
        "unitVectors": int(unit.group(1)), "unitMismatches": int(unit.group(2)),
        "wrapperInstructions": int(acc.group(1)), "wrapperIdleCycles": int(acc.group(2)),
        "wrapperMismatches": int(acc.group(3)) + int(rd.group(2)),
    }


def soc(isa_text, mnist_text):
    """PicoRV32 SoC RTL simulation at 0 wait states: self-test, MNIST outputs, cycles per inference."""
    ints = lambda pat: int(re.search(pat, mnist_text).group(1))
    images = [l for l in mnist_text.splitlines() if l.startswith("test image")]
    return {
        "isaChecks": int(re.search(r"checks: (\d+)", isa_text).group(1)),
        "isaPass": "RESULT: all custom-0 instructions match the C reference" in isa_text,
        "imagesChecked": len(images),
        "imagesCorrect": sum(1 for l in images if re.search(r"label (\d+), predicted \1,", l)),
        "allMatch": "RESULT: all outputs match" in mnist_text,
        "instructionsPerInference": ints(r"one inference: (\d+) instructions"),
        "macClrAddInstructionsPerInference": ints(r"with simd_mac_clr \+ add: (\d+) instructions"),
        "macInstructionsPerInference": ints(r"with simd_macz/simd_macc: (\d+) instructions"),
        "optScalarInstructionsPerInference": ints(r"optimised scalar \(unrolled\): (\d+) instructions"),
        "optMacInstructionsPerInference": ints(r"optimised MAC \(unrolled simd_macz/simd_macc\): (\d+) instructions"),
        "scalarCycles": ints(r"cycles per inference: scalar (\d+)"),
        "macClrAddCycles": ints(r"simd_mac_clr \+ add (\d+) \("),
        "macCycles": ints(r"simd_macz/simd_macc (\d+) \("),
        "optScalarCycles": ints(r"optimised scalar \(unrolled\): \d+ instructions, (\d+) cycles"),
        "optMacCycles": ints(r"optimised MAC \(unrolled simd_macz/simd_macc\): \d+ instructions, (\d+) cycles"),
    }


def sweep(text):
    cols = ("waitStates", "scalar", "macClrAdd", "macInternal", "optScalar", "optMac")
    rows = []
    for m in re.finditer(r"^(\d+) (\d+) (\d+) (\d+) (\d+) (\d+) ([0-9.]+)$", text, re.M):
        row = dict(zip(cols, map(int, m.groups()[:6])))
        row["speedup"] = float(m.group(7))
        rows.append(row)
    base = re.search(r"one inference: (\d+) instructions, (\d+) cycles", text)
    return {"rows": rows, "noMacScalarCycles": int(base.group(2))}


def area(text):
    """Yosys/nextpnr numbers for the PicoRV32 SoC, from hardware/scripts/soc_synth.sh."""
    out, family, current = {}, None, None
    for line in text.splitlines():
        if line.startswith("== "):
            family = "ecp5" if "ECP5" in line else "ice40"
            out[family] = {}
            continue
        m = re.match(r"(core, no MAC unit|core \+ MAC unit|MAC unit alone)\s+LUT4\s+(\d+)\s+FF\s+(\d+)\s+DSP\s+(\d+)", line)
        if m and family:
            key = {"core, no MAC unit": "coreNoMac", "core + MAC unit": "coreMac", "MAC unit alone": "unit"}[m.group(1)]
            current = {"lut4": int(m.group(2)), "ff": int(m.group(3)), "dsp": int(m.group(4))}
            out[family][key] = current
            continue
        m = re.match(r"\s+seed \d+: ([0-9.]+) MHz", line)
        if m and current is not None:
            current.setdefault("fmaxMHz", []).append(float(m.group(1)))
    return out


INSTRUCTION_KEYS = ("instructionsPerInference", "macClrAddInstructionsPerInference", "macInstructionsPerInference",
                    "optScalarInstructionsPerInference", "optMacInstructionsPerInference")


def fmax(text):
    out = {}
    for design, fpga, mhz in re.findall(r"^(unit|wrapper) (\S+) seed \d+: ([0-9.]+) MHz", text, re.M):
        out.setdefault(f"{design} {fpga}", []).append(float(mhz))
    return {k: {"min": min(v), "max": max(v), "seeds": len(v)} for k, v in out.items()}


def load_committed():
    js = read(OUT)
    if not js:
        return None
    return json.loads(js.split("window.PROJECT_RESULTS = ", 1)[1].rsplit(";", 1)[0])


def main():
    fw_text, cs_text = read("build/firmware_run.txt"), read("build/cosim.txt")
    committed = load_committed() or {}
    soc_isa, soc_mnist = read("build/soc_isa.txt"), read("build/soc_mnist.txt")
    results = {}
    if fw_text is not None:
        results["firmware"] = firmware(fw_text)
    if cs_text is not None:
        results["cosim"] = cosim(cs_text)
    if soc_isa is not None and soc_mnist is not None:
        results["soc"] = soc(soc_isa, soc_mnist)
    sw_text, ar_text = read("build/soc_sweep.txt"), read("build/soc_synth.txt")
    if sw_text is not None:
        results["socSweep"] = sweep(sw_text)
    if ar_text is not None:
        results["socArea"] = area(ar_text)

    if "--check" in sys.argv:
        if not results:
            sys.exit("nothing to check: run the firmware or the co-simulation first")
        for key in results:
            if key in ("socSweep", "socArea"):
                continue
            if committed.get(key) != results[key]:
                print(f"web/js/results.js is out of date ({key}). Run scripts/record_results.py and commit it.")
                print("committed:", json.dumps(committed.get(key)))
                print("measured: ", json.dumps(results[key]))
                sys.exit(1)
        if "soc" in results:
            fw = results.get("firmware") or committed.get("firmware") or {}
            for k in INSTRUCTION_KEYS:
                if results["soc"][k] != fw.get(k):
                    print(f"RTL and Spike disagree on {k}: RTL {results['soc'][k]}, Spike {fw.get(k)}")
                    sys.exit(1)
            if not (results["soc"]["isaPass"] and results["soc"]["allMatch"]):
                print("the SoC simulation did not report a clean pass")
                sys.exit(1)
        soc_now = results.get("soc") or committed.get("soc")
        sw = results.get("socSweep") or committed.get("socSweep")
        if soc_now and sw:
            zero = next(r for r in sw["rows"] if r["waitStates"] == 0)
            for a, b in (("scalar", "scalarCycles"), ("macClrAdd", "macClrAddCycles"), ("macInternal", "macCycles"),
                         ("optScalar", "optScalarCycles"), ("optMac", "optMacCycles")):
                if zero[a] != soc_now[b]:
                    print(f"the 0-wait-state sweep row and the SoC run disagree on {b}: {zero[a]} against {soc_now[b]}")
                    sys.exit(1)
        print("web/js/results.js matches the measured results:", ", ".join(results))
        return

    if fw_text is None or cs_text is None:
        sys.exit("run the firmware and the co-simulation first (see the docstring for the commands)")
    fm_text = read("build/fmax.txt")
    for key in ("soc", "socSweep", "socArea"):
        if key not in results and key in committed:
            results[key] = committed[key]
    results["fmax"] = fmax(fm_text) if fm_text else committed.get("fmax", {})
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("// Measured results shown on the web demo. Generated by scripts/record_results.py.\n")
        f.write("window.PROJECT_RESULTS = " + json.dumps(results, indent=2) + ";\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
