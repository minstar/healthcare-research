#!/usr/bin/env python3
"""Scatter: MedQA accuracy vs OpenBioRQ frozen-core solve rate (v2, clean style).

Closed-form MedQA saturates into a narrow band while OpenBioRQ spreads across the full range;
within the open-weight tier the axes barely track, and frontier agents climb both.
Saved to paper_writing/figures/fig_orthogonality.pdf  (run with the torchtitan env's python).
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "paper_writing"))
import figstyle as S
S.apply()

R = os.path.join(ROOT, "results")
MO = f"{R}/medqa_ortho"

def medqa(tag):
    return json.load(open(f"{MO}/{tag}_summary.json"))["per_dataset"]["MedQA"]["acc"]

lb = {e["model"]: e for e in json.load(open(f"{R}/leaderboard_core_t0.json"))["leaderboard"]}
def robust(name):
    return round(100 * lb[name]["robust_core"]["solve@0.5"], 1)

_fr = json.load(open(f"{R}/frontier_robust_leaderboard.json"))
FRONTIER_ROBUST = {
    "Gemini-3-Pro": _fr["api_gemini3pro_robust"]["solve@0.5"],
    "Opus-4.7": _fr["api_opus47_robust"]["solve@0.5"],
    "GPT-5.5": _fr["api_gpt55_robust"]["solve@0.5"],
}

tiers = {
    "Roster (defines the core)": [("GLM-5.1", "glm51", 0.0), ("Qwen3.6", "qwen36", 0.0),
                                  ("DeepSeek-V4", "dsv4", 0.0)],
    "Held-out (same lineage)": [("GLM-5", "glm5", robust("GLM-5")),
                                ("Qwen3.5-397B", "qwen35_397b", robust("Qwen3.5-397B")),
                                ("Qwen3-235B", "qwen3_235b", robust("Qwen3-235B"))],
    "Frontier (independent lineage)": [("Gemini-3-Pro", "gemini3pro", FRONTIER_ROBUST["Gemini-3-Pro"]),
                                       ("Opus-4.7", "opus47", FRONTIER_ROBUST["Opus-4.7"]),
                                       ("GPT-5.5", "gpt55", FRONTIER_ROBUST["GPT-5.5"])],
}
colors = {"Roster (defines the core)": S.GREY, "Held-out (same lineage)": S.BLUE,
          "Frontier (independent lineage)": S.CORAL}
markers = {"Roster (defines the core)": "s", "Held-out (same lineage)": "o",
           "Frontier (independent lineage)": "^"}

fig, ax = plt.subplots(figsize=(5.6, 3.9))
xs_all = []
for tier, items in tiers.items():
    xs, ys = [], []
    is_roster = tier.startswith("Roster")
    for name, tag, y in items:
        x = medqa(tag)
        xs.append(x); ys.append(y); xs_all.append(x)
        if not is_roster:
            ax.annotate(name, (x, y), fontsize=6.8, xytext=(4, 4),
                        textcoords="offset points", color=S.INK)
    ax.scatter(xs, ys, c=colors[tier], marker=markers[tier], s=62,
               edgecolors="#3A3F46", linewidths=0.5, zorder=3)
    if is_roster:
        ax.annotate("roster (= 0 by construction)", (sum(xs) / len(xs), 0), fontsize=6.8,
                    xytext=(0, -13), textcoords="offset points", ha="center", color=S.INK2)

# MedQA band shading (the saturation story)
lo, hi = min(xs_all), max(xs_all)
ax.axvspan(lo, hi, color=S.AMBER, alpha=0.12, zorder=0)
ax.text((lo + hi) / 2, 64, f"MedQA band: {lo:.0f}-{hi:.0f}%  ({hi-lo:.0f} pt)", ha="center",
        va="top", fontsize=7.2, color="#B07E2E", fontweight="bold")

ax.set_xlabel("MedQA-USMLE accuracy (%)")
ax.set_ylabel("OpenBioRQ frozen-core solve rate (%)")
ax.set_ylim(-5, 70); ax.set_xlim(lo - 2.5, hi + 2.5)
ax.grid(True, color=S.HAIR, linewidth=0.9, zorder=0)
S.title(ax, "Closed-form MedQA saturates; OpenBioRQ spreads")

leg = [Line2D([0], [0], marker=markers[t], color="w", markerfacecolor=colors[t],
              markeredgecolor="#3A3F46", markersize=8, label=t) for t in tiers]
ax.legend(handles=leg, fontsize=7.0, loc="upper left", borderaxespad=0.6,
          handletextpad=0.5, labelspacing=0.5)

fig.tight_layout()
out = os.path.join(ROOT, "paper_writing/figures/fig_orthogonality.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
