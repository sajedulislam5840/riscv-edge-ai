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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = load_js_object(ROOT / "web" / "js" / "results.js")
    trace = load_js_object(ROOT / "web" / "js" / "trace.js")
    fig_versions(results)
    fig_waitstates(results)
    fig_neuron(trace)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))


if __name__ == "__main__":
    main()
