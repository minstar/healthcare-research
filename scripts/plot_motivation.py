#!/usr/bin/env python3
"""Motivation figure: positioning OpenBioRQ vs prior biomedical benchmarks, including Anthropic's
recent agentic BioMysteryBench. Two axes: tool-using/agentic (x) vs answer status (y, objective/
verifiable answer -> genuinely open/no answer key). The entire field---including BioMysteryBench,
which by design uses questions with objective verifiable answers---sits below the "open" line;
OpenBioRQ is the only one in the agentic + no-answer-key regime, where wrong-paper citation,
agentic collapse, and abstention become measurable.
Saved to paper_writing/figures/fig_motivation.pdf  (run with the torchtitan env python).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Shared house style (consistent with the other OpenBioRQ figures; fonttype 42 avoids
# Type-3 fonts that AAAI rejects).
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.linewidth": 0.8, "axes.edgecolor": "#444444",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "legend.frameon": False, "figure.dpi": 150,
})

# (name, x agentic 0-10, y answer-open 0-10, group)
PTS = [
    ("MedQA", 0.9, 1.7, "closed"),
    ("PubMedQA", 1.3, 2.6, "closed"),
    ("MedMCQA", 2.1, 0.8, "closed"),
    ("BioASQ", 2.9, 1.9, "closed"),
    ("GPQA", 3.1, 3.2, "closed"),
    ("LAB-Bench", 4.2, 3.5, "closed"),
    ("GeneGPT", 5.4, 2.3, "agentic_known"),
    ("BioMysteryBench\n(Anthropic '26)", 8.1, 3.3, "agentic_known"),
    ("OpenBioRQ\n(ours)", 8.6, 8.4, "open"),
]
COL = {"closed": "#9aa0a6", "agentic_known": "#e8821a", "open": "#d62728"}
MARK = {"closed": "o", "agentic_known": "s", "open": "*"}

fig, ax = plt.subplots(figsize=(6.4, 5.0))
# quadrant guides
ax.axhline(5, color="black", lw=1.0, ls="--", alpha=0.6)
ax.axvline(5, color="black", lw=0.8, ls=":", alpha=0.4)
# highlight the open+agentic quadrant
ax.add_patch(plt.Rectangle((5, 5), 5, 5, color="#d62728", alpha=0.06, zorder=0))
ax.text(7.5, 9.55, "agentic  +  no answer key", ha="center", fontsize=8.5,
        color="#d62728", weight="bold")
ax.text(7.5, 4.35, "agentic, objective/verifiable answer", ha="center", fontsize=7.5, color="#b5660a")
ax.text(2.6, 4.35, "static closed-form QA", ha="center", fontsize=7.5, color="#666666")

for name, x, y, g in PTS:
    big = g == "open"
    ax.scatter([x], [y], s=320 if big else 90, c=COL[g], marker=MARK[g],
               edgecolors="black", linewidths=0.7, zorder=3)
    dy = 0.42 if not big else 0.0
    ax.annotate(name, (x, y), fontsize=7.5 if not big else 8.5,
                weight="bold" if big else "normal",
                xytext=(0, 11 if big else 8), textcoords="offset points", ha="center")

# callout: what becomes measurable only in the open+agentic regime
ax.annotate("wrong-paper citation,\nagentic collapse, abstention\nmeasurable only here",
            xy=(8.6, 8.4), xytext=(5.4, 6.5), fontsize=7.2, color="#d62728",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))

ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_xlabel("Tool-using / agentic  $\\rightarrow$", fontsize=10)
ax.set_ylabel("Answer status:  objective/known  $\\rightarrow$  genuinely open", fontsize=10)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Even agentic biomedical benchmarks keep a known answer;\nOpenBioRQ targets unsolved questions",
             fontsize=10.5)
# legend
from matplotlib.lines import Line2D
leg = [Line2D([0],[0],marker='o',color='w',markerfacecolor=COL['closed'],markeredgecolor='k',markersize=8,label='closed-form medical QA'),
       Line2D([0],[0],marker='s',color='w',markerfacecolor=COL['agentic_known'],markeredgecolor='k',markersize=8,label='agentic, objective answer'),
       Line2D([0],[0],marker='*',color='w',markerfacecolor=COL['open'],markeredgecolor='k',markersize=15,label='agentic, open question (ours)')]
ax.legend(handles=leg, fontsize=7.2, loc="upper left", framealpha=0.92)
fig.tight_layout()
out = "/data/project/private/minstar/workspace/healthcare-research/paper_writing/figures/fig_motivation.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
