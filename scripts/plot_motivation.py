#!/usr/bin/env python3
"""Motivation figure (v2, clean style): positioning OpenBioRQ vs prior biomedical benchmarks.
Axes: tool-using/agentic (x) vs answer status (y: objective/known -> genuinely open). The whole
field---including Anthropic's agentic BioMysteryBench---sits in the known-answer band; OpenBioRQ is
the only one in the agentic + no-answer-key regime, where wrong-paper citation, agentic collapse,
and abstention become measurable.
Saved to paper_writing/figures/fig_motivation.pdf  (run with the torchtitan env python).
"""
import sys, os
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "paper_writing"))
import figstyle as S
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

S.apply()

# (name, x agentic 0-10, y answer-open 0-10, group)
PTS = [
    ("MedQA", 0.9, 1.7, "closed"),
    ("PubMedQA", 1.4, 2.7, "closed"),
    ("MedMCQA", 2.2, 0.9, "closed"),
    ("BioASQ", 3.0, 2.0, "closed"),
    ("GPQA", 3.2, 3.3, "closed"),
    ("LAB-Bench", 4.3, 3.6, "closed"),
    ("MedLFQA / OLAPH", 1.8, 4.2, "longform"),
    ("HealthBench", 3.9, 4.6, "longform"),
    ("GeneGPT", 5.5, 2.4, "agentic_known"),
    ("BioMysteryBench\n(Anthropic '26)", 8.0, 3.4, "agentic_known"),
    ("OpenBioRQ\n(ours)", 8.4, 7.9, "open"),
]
COL = {"closed": S.GREY, "longform": S.TEAL, "agentic_known": S.AMBER, "open": S.CORAL}
MARK = {"closed": "o", "longform": "D", "agentic_known": "s", "open": "*"}
LBL = {"closed": "closed-form medical QA",
       "longform": "long-form answer, known reference",
       "agentic_known": "agentic, objective answer",
       "open": "agentic, open question (ours)"}

fig, ax = plt.subplots(figsize=(6.6, 4.9))

# soft-tinted target quadrant (agentic + open) — the only thing the eye should land on
ax.add_patch(plt.Rectangle((5, 5), 5, 5, facecolor=S.CORAL, alpha=0.07,
                           edgecolor="none", zorder=0))
# faint quadrant dividers
ax.axhline(5, color="#D7DAE0", lw=0.9, ls=(0, (5, 4)), zorder=1)
ax.axvline(5, color="#D7DAE0", lw=0.9, ls=(0, (5, 4)), zorder=1)

# quadrant region labels (muted)
ax.text(5.25, 9.6, "agentic  $\\times$  no answer key", ha="left", va="top",
        fontsize=8.5, color=S.CORAL, fontweight="bold")
ax.text(7.5, 0.45, "agentic, verifiable answer", ha="center", va="bottom",
        fontsize=7.6, color=S.INK2)
ax.text(2.5, 0.45, "static QA, known answer", ha="center", va="bottom",
        fontsize=7.6, color=S.INK2)

# scatter
for name, x, y, g in PTS:
    big = g == "open"
    ax.scatter([x], [y], s=440 if big else 95, c=COL[g], marker=MARK[g],
               edgecolors=("#B4451F" if big else "#3A3F46"),
               linewidths=(1.1 if big else 0.7), zorder=4)
    ax.annotate(name, (x, y), fontsize=(9.0 if big else 7.6),
                color=(S.CORAL if big else S.INK),
                fontweight=("bold" if big else "normal"),
                xytext=(0, 14 if big else 8), textcoords="offset points", ha="center",
                linespacing=1.05, zorder=5)

# --- relationship cues (restrained: lineage in, contrast on the right) ---
# (1) rubric lineage: OpenBioRQ's per-question checklist descends from the
#     reference must-have / nice-to-have rubrics of MedLFQA/OLAPH & HealthBench.
ax.add_patch(FancyArrowPatch((3.20, 4.78), (7.62, 7.35),
             connectionstyle="arc3,rad=-0.40", arrowstyle="-|>",
             mutation_scale=12, color=S.TEAL, lw=1.05,
             linestyle=(0, (1, 2.2)), alpha=0.9, zorder=3))
ax.text(4.65, 7.62, "shared checklist-rubric\nlineage", color=S.TEAL,
        fontsize=7.5, ha="center", va="center", style="italic",
        linespacing=1.12, zorder=5)

# (2) BioMysteryBench is the mirror coordinate: same agentic axis, opposite
#     answer status (verifiable by design vs. no answer key).
ax.add_patch(FancyArrowPatch((7.98, 3.78), (8.34, 7.42),
             arrowstyle="<|-|>", mutation_scale=11, color=S.INK2,
             lw=1.05, linestyle=(0, (4, 3)), zorder=3))
ax.text(8.52, 5.55, "opposite\nanswer status", color=S.INK2, fontsize=7.5,
        ha="left", va="center", linespacing=1.15, zorder=5)

# (3) the takeaway (no swooping arrow — the star already sits in the quadrant)
ax.text(5.05, 5.85, "only here:\ngrounding, abstention, &\nwrong-paper citation\nbecome measurable",
        fontsize=7.3, color=S.CORAL, ha="left", va="center",
        linespacing=1.28, zorder=5)

ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_xlabel("Tool-using / agentic  $\\rightarrow$", fontsize=9.5)
ax.set_ylabel("Answer status:  objective / known  $\\rightarrow$  genuinely open", fontsize=9.5)
ax.set_xticks([]); ax.set_yticks([])
for s in ("left", "bottom"):
    ax.spines[s].set_color(S.SPINE)

# clean bold title (no colored box)
ax.set_title("Even agentic biomedical benchmarks keep a known answer;\n"
             "OpenBioRQ targets unsolved questions",
             loc="center", fontsize=10.5, fontweight="bold", color=S.INK, pad=10)

# legend (clean, upper-left, inside the empty open-y / static-x corner)
leg = [Line2D([0], [0], marker=MARK[g], color="w", markerfacecolor=COL[g],
              markeredgecolor="#3A3F46", markersize=(13 if g == "open" else 8),
              label=LBL[g]) for g in ("closed", "longform", "agentic_known", "open")]
ax.legend(handles=leg, fontsize=7.4, loc="upper left", borderaxespad=0.6,
          handletextpad=0.5, labelspacing=0.5)

fig.tight_layout()
out = os.path.join(ROOT, "paper_writing/figures/fig_motivation.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
