#!/usr/bin/env python3
"""Scatter: MedQA accuracy vs OpenBioRQ robust-core pass@0.5 (Reviewer E figure).

Shows closed-form MedQA saturating into a narrow band while OpenBioRQ spreads across the full
range -- within the open-weight tier the axes are orthogonal; frontier agents climb both.
Saved to paper_writing/figures/fig_orthogonality.pdf  (run with the torchtitan env's python).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/data/project/private/minstar/workspace/healthcare-research/results"
MO = f"{R}/medqa_ortho"

def medqa(tag):
    return json.load(open(f"{MO}/{tag}_summary.json"))["per_dataset"]["MedQA"]["acc"]

lb = {e["model"]: e for e in json.load(open(f"{R}/leaderboard_core_t0.json"))["leaderboard"]}
def robust(name):
    return round(100 * lb[name]["robust_core"]["solve@0.5"], 1)

fr = json.load(open(f"{R}/frontier_robust_leaderboard.json")) if False else None
FRONTIER_ROBUST = {"Gemini-3-Pro": 28.8, "Opus-4.7": 37.8, "GPT-5.5": 59.6}

# tier -> [(model, medqa_tag, y_robust)]
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
colors = {"Roster (defines the core)": "#888888", "Held-out (same lineage)": "#1f77b4",
          "Frontier (independent lineage)": "#d62728"}
markers = {"Roster (defines the core)": "s", "Held-out (same lineage)": "o",
           "Frontier (independent lineage)": "^"}

fig, ax = plt.subplots(figsize=(5.4, 3.8))
xs_all = []
for tier, items in tiers.items():
    xs, ys = [], []
    is_roster = tier.startswith("Roster")
    for name, tag, y in items:
        x = medqa(tag)
        xs.append(x); ys.append(y); xs_all.append(x)
        if not is_roster:  # roster points collide at y=0; label the group once instead
            ax.annotate(name, (x, y), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.scatter(xs, ys, c=colors[tier], marker=markers[tier], s=55, label=tier,
               edgecolors="black", linewidths=0.4, zorder=3)
    if is_roster:
        ax.annotate("roster (=0 by construction)", (sum(xs) / len(xs), 0), fontsize=6.5,
                    xytext=(0, -12), textcoords="offset points", ha="center", color="#555555")

# MedQA band shading
lo, hi = min(xs_all), max(xs_all)
ax.axvspan(lo, hi, color="orange", alpha=0.08, zorder=0)
ax.text((lo + hi) / 2, 62, f"MedQA band\n{lo:.0f}-{hi:.0f}% ({hi-lo:.0f} pt)", ha="center",
        fontsize=7, color="darkorange")

ax.set_xlabel("MedQA-USMLE accuracy (%)")
ax.set_ylabel("OpenBioRQ robust-core pass@0.5 (%)")
ax.set_ylim(-4, 68)
ax.set_xlim(lo - 2.5, hi + 2.5)
ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.25, zorder=0)
ax.set_title("Closed-form MedQA saturates; OpenBioRQ spreads", fontsize=9)
fig.tight_layout()
out = "/data/project/private/minstar/workspace/healthcare-research/paper_writing/figures/fig_orthogonality.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
# also print the data used
for tier, items in tiers.items():
    for name, tag, y in items:
        print(f"  {name:14s} MedQA={medqa(tag):.1f}  robust={y}")
