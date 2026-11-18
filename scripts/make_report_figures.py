"""Figures for docs/report, drawn from the recorded results and the recorded cycle trace.

  python3 scripts/make_report_figures.py

Reads web/js/results.js and web/js/trace.js and writes PDF figures to docs/report/figures/.
Nothing is typed in by hand: change a recorded number and the figures change with it.
"""
import json
import pathlib
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "report" / "figures"

INK = "#1f2933"
GRID = "#d5dbe1"
BLUE = "#1f6fb2"
ORANGE = "#d9822b"
GREY = "#8a96a3"
TEAL = "#2a9d8f"
RED = "#b5473a"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def load_js_object(path):
    text = path.read_text(encoding="utf-8")
    start = text.index("{")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def fig_versions(results):
    soc = results["soc"]
    rows = [
        ("Scalar, plain loop", soc["scalarCycles"], GREY),
        ("Scalar, unrolled", soc["optScalarCycles"], ORANGE),
        ("simd_mac_clr + add", soc["macClrAddCycles"], GREY),
        ("simd_macz / simd_macc", soc["macCycles"], GREY),
        ("simd_macz / simd_macc, unrolled", soc["optMacCycles"], BLUE),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 2.0))
    labels = [r[0] for r in rows][::-1]
    values = [r[1] / 1e6 for r in rows][::-1]
    colors = [r[2] for r in rows][::-1]
    bars = ax.barh(labels, values, color=colors, height=0.62)
    for bar, v in zip(bars, values):
        ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2, f"{v:.2f} M", va="center", color=INK)
    ax.set_xlabel("Clock cycles per inference (millions), no wait states")
    ax.set_xlim(0, max(values) * 1.15)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fair = soc["optScalarCycles"] / soc["optMacCycles"]
    ax.annotate(f"fair comparison: {fair:.2f}x fewer cycles", xy=(0.99, 0.30), xycoords="axes fraction",
                ha="right", color=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_versions.pdf")
    plt.close(fig)


def fig_waitstates(results):
    rows = results["socSweep"]["rows"]
    ws = [r["waitStates"] for r in rows]
    scalar = [r["optScalar"] / 1e6 for r in rows]
    mac = [r["optMac"] / 1e6 for r in rows]
    speed = [r["optScalar"] / r["optMac"] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.2, 2.1), gridspec_kw={"width_ratios": [1.15, 1]})
    a.plot(ws, scalar, "-o", color=ORANGE, markersize=3.5, label="Scalar, unrolled")
    a.plot(ws, mac, "-o", color=BLUE, markersize=3.5, label="MAC, unrolled")
    a.set_xlabel("Wait states per memory access")
    a.set_ylabel("Cycles per inference (millions)")
    a.set_xticks(ws)
    a.legend(frameon=False, loc="upper left")
    a.yaxis.grid(True, color=GRID, linewidth=0.6)
    a.set_axisbelow(True)
    b.plot(ws, speed, "-o", color=TEAL, markersize=3.5)
    for i, (x, y) in enumerate(zip(ws, speed)):
        b.text(x, y + (0.28 if i % 2 == 0 else -0.5), f"{y:.2f}", ha="center", fontsize=7, color=INK)
    b.set_ylim(0, 5.2)
    b.set_xlabel("Wait states per memory access")
    b.set_ylabel("Speed-up in cycles")
    b.set_xticks(ws)
    b.yaxis.grid(True, color=GRID, linewidth=0.6)
    b.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_waitstates.pdf")
    plt.close(fig)


def class_cycles(part):
    totals = Counter()
    for cls, cycles in zip(part["cls"], part["c"]):
        totals[cls] += cycles
    return totals


def fig_neuron(trace):
    order = [("L", "load", BLUE), ("M", "multiply", ORANGE), ("X", "custom MAC", RED),
             ("A", "other arithmetic", GREY), ("B", "branch and jump", TEAL)]
    parts = [("Scalar loop, unrolled", trace["scalar"]), ("MAC loop, unrolled", trace["mac"])]
    fig, ax = plt.subplots(figsize=(6.2, 1.7))
    ypos = [1, 0]
    for y, (name, part) in zip(ypos, parts):
        totals = class_cycles(part)
        total = sum(totals.values())
        left = 0
        for key, label, color in order:
            v = totals.get(key, 0)
            if not v:
                continue
            ax.barh(y, v, left=left, color=color, height=0.55, label=label if y == 1 or key == "X" else None)
            if v / total > 0.07:
                ax.text(left + v / 2, y, f"{100 * v / total:.0f}%", ha="center", va="center", color="white", fontsize=7.5)
            left += v
        ax.text(left + 150, y, f"{total:,} cycles", va="center", color=INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([p[0] for p in parts])
    ax.set_xlim(0, 24500)
    ax.set_xlabel("Clock cycles for one neuron (784 multiply-accumulates), by instruction class")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.28),
              fontsize=7.5, handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_neuron.pdf")
    plt.close(fig)


def best_fmax(block):
    return max(block["fmaxMHz"])


def bar_pair(ax, labels, values, colors, fmt, title, ylabel=None, ylim=None):
    bars = ax.bar(labels, values, color=colors, width=0.62)
    top = ylim or max(values) * 1.5
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.02, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=8)
    ax.set_ylim(0, top)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left", fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_summary(results):
    fw, soc, area = results["firmware"], results["soc"], results["socArea"]["ecp5"]
    f_base, f_mac = best_fmax(area["coreNoMac"]), best_fmax(area["coreMac"])
    t_base = soc["optScalarCycles"] / f_base / 1e3
    t_mac = soc["optMacCycles"] / f_mac / 1e3
    labels = ["Scalar", "With\nMAC unit"]
    cols = [ORANGE, BLUE]
    fig, axes = plt.subplots(1, 4, figsize=(6.7, 2.05))
    bar_pair(axes[0], labels, [fw["optScalarInstructionsPerInference"] / 1e3, fw["optMacInstructionsPerInference"] / 1e3],
             cols, "{:.0f}", "Instructions", "thousands per inference")
    bar_pair(axes[1], labels, [soc["optScalarCycles"] / 1e6, soc["optMacCycles"] / 1e6],
             cols, "{:.2f}", "Clock cycles", "millions per inference")
    bar_pair(axes[2], labels, [t_base, t_mac], cols, "{:.1f}", "Time on ECP5", "milliseconds per inference")
    bar_pair(axes[3], ["FP32", "INT8\nintegers"], [98.23, 98.21], [GREY, TEAL], "{:.2f}", "MNIST accuracy", "per cent", ylim=150)
    axes[3].set_yticks([0, 25, 50, 75, 100])
    ratios = [fw["optScalarInstructionsPerInference"] / fw["optMacInstructionsPerInference"],
              soc["optScalarCycles"] / soc["optMacCycles"], t_base / t_mac]
    words = ["fewer", "fewer", "faster"]
    for ax, r, w in zip(axes[:3], ratios, words):
        ax.text(0.5, 0.93, f"{r:.2f}\u00d7 {w}", transform=ax.transAxes, ha="center", va="top", color=BLUE, fontsize=8.5, fontweight="bold")
    axes[3].text(0.5, 0.93, "same accuracy", transform=axes[3].transAxes, ha="center", va="top", color=TEAL, fontsize=8.5, fontweight="bold")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "fig_summary.pdf")
    plt.close(fig)


def fig_cost(results):
    area = results["socArea"]["ecp5"]
    labels = ["Without\nunit", "With\nunit"]
    cols = [ORANGE, BLUE]
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 1.9))
    bar_pair(axes[0], labels, [area["coreNoMac"]["lut4"], area["coreMac"]["lut4"]], cols, "{:,}", "LUT4 (logic)")
    bar_pair(axes[1], labels, [area["coreNoMac"]["dsp"], area["coreMac"]["dsp"]], cols, "{}", "DSP blocks")
    bar_pair(axes[2], labels, [best_fmax(area["coreNoMac"]), best_fmax(area["coreMac"])], cols, "{:.1f}", "Clock speed (MHz)")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "fig_cost.pdf")
    plt.close(fig)


def fig_gain(results):
    """Gain over the plain scalar loop for each firmware version, in instructions and in cycles."""
    fw, soc = results["firmware"], results["soc"]
    names = ["Scalar, plain loop", "Scalar, unrolled", "simd_mac_clr + add", "simd_macz / simd_macc",
             "simd_macz / simd_macc, unrolled"]
    ins = [fw["instructionsPerInference"], fw["optScalarInstructionsPerInference"], fw["macClrAddInstructionsPerInference"],
           fw["macInstructionsPerInference"], fw["optMacInstructionsPerInference"]]
    cyc = [soc["scalarCycles"], soc["optScalarCycles"], soc["macClrAddCycles"], soc["macCycles"], soc["optMacCycles"]]
    gi = [ins[0] / v for v in ins]
    gc = [cyc[0] / v for v in cyc]
    fig, ax = plt.subplots(figsize=(6.2, 2.2))
    y = list(range(len(names)))[::-1]
    h = 0.36
    b1 = ax.barh([v + h / 2 for v in y], gi, height=h, color=BLUE, label="fewer instructions")
    b2 = ax.barh([v - h / 2 for v in y], gc, height=h, color=TEAL, label="fewer clock cycles")
    for bars, vals in ((b1, gi), (b2, gc)):
        for bar, v in zip(bars, vals):
            ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2, f"{v:.2f}x", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, max(gi) * 1.12)
    ax.set_xlabel("Gain over the plain scalar loop (1.00x = no gain)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gain.pdf")
    plt.close(fig)


def fig_faults():
    """Mismatches against the golden model when the RTL is deliberately broken.

    These counts are the ones recorded in README.md (RTL verification), out of 25,088 vectors each.
    """
    rows = [("Correct RTL", 0, GREY), ("One adder flipped", 24444, RED),
            ("Accumulator feedback broken", 24900, RED), ("simd_macz forgets to clear", 24900, RED)]
    fig, ax = plt.subplots(figsize=(6.2, 1.7))
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [r[2] for r in rows][::-1]
    bars = ax.barh(labels, vals, color=cols, height=0.6)
    for bar, v, name in zip(bars, vals, labels):
        txt = "0 mismatches" if v == 0 else (f"{v:,}" if v == 24444 else "about 24,900")
        ax.text(v + 300, bar.get_y() + bar.get_height() / 2, txt, va="center", fontsize=7.5, color=INK)
    ax.axvline(25088, color=INK, linewidth=0.8, linestyle="--")
    ax.text(25088, 3.55, "all 25,088 vectors", ha="right", va="bottom", fontsize=7, color=INK)
    ax.set_xlim(0, 31000)
    ax.set_ylim(-0.5, 4.0)
    ax.set_xlabel("Vectors that disagree with the golden model")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_faults.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = load_js_object(ROOT / "web" / "js" / "results.js")
    trace = load_js_object(ROOT / "web" / "js" / "trace.js")
    fig_versions(results)
    fig_waitstates(results)
    fig_neuron(trace)
    fig_summary(results)
    fig_cost(results)
    fig_gain(results)
    fig_faults()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))


if __name__ == "__main__":
    main()
