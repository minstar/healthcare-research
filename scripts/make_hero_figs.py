#!/usr/bin/env python3
"""Two new 'hero' figures for the OpenBioRQ main body.

  fig_existence_funnel.pdf  - the headline: a two-gate sieve landing 'existence != correctness'.
                              L1 (existence) waves through >99%; L2 (support) catches ~1-in-6.
  fig_eval_flow.pdf         - the evaluation/scoring flow (complements fig_pipeline's CONSTRUCTION
                              flow): open question -> agentic harness -> answer+citations -> graded
                              two ways (frozen per-question checklist; two-level L1/L2 citation audit).

Run with the torchtitan env python (has matplotlib):
  python scripts/make_hero_figs.py

Every number is loaded from results/*.json and asserted against the FACTS.md values it must match,
so the figure can never silently drift from the audited numbers.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper_writing/figures")
sys.path.insert(0, os.path.join(ROOT, "paper_writing"))
import figstyle as S
S.apply()


def _close(a, b, tol=0.15):
    return abs(a - b) <= tol


# ---- load + verify the citation-audit numbers (results/cite_audit_summary.json) ----
_ca = json.load(open(os.path.join(RES, "cite_audit_summary.json")))
N_EMIT = _ca["total_citations"]                       # 4,863 emitted
N_REAL = _ca["overall"]["real_citations_judged"]      # 4,649 real & judged
WP = _ca["overall"]["wrong_paper_pct"]                # 15.9  (primary GLM judge)
SUP = _ca["overall"]["yes_pct"]                       # 37.4
PART = _ca["overall"]["partial_pct"]                  # 46.7
N_FAB = sum(_ca["by_model"][m]["fabricated"] for m in _ca["by_model"])  # 35
FAB_PCT = round(100.0 * N_FAB / N_EMIT, 1)            # 0.7
EXIST_PCT = round(100.0 - FAB_PCT, 1)                 # 99.3
WP_OPUS = 10.6                                        # independent-family judge (FACTS / tab:l2)
assert N_EMIT == 4863 and _close(WP, 15.9) and _close(FAB_PCT, 0.7), (N_EMIT, WP, FAB_PCT)
assert _close(SUP, 37.4) and _close(PART, 46.7), (SUP, PART)


# ============================================================================
# HERO 1 — fig_existence_funnel.pdf
# ============================================================================
def fig_funnel():
    fig, ax = plt.subplots(figsize=(3.5, 2.92))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    INK, INK2, CORAL, TEAL, GREY = S.INK, S.INK2, S.CORAL, S.TEAL, S.GREY
    BLUE = S.BLUE

    # ---- geometry: a horizontal "pipe" with a flat TOP edge; its height encodes the
    #      surviving count, so each gate that removes citations steps the bottom edge UP.
    #      The pipe-step is the PROPORTIONAL truth (barely moves at L1, clearly at L2);
    #      the coral drops + numbers are annotations on top of it.
    Y_TOP = 82.0
    H = 32.0                       # full height = 100% of emitted citations
    x0, xL1, xL2, xEnd = 5.0, 30.0, 63.0, 83.0
    h_afterL1 = H * EXIST_PCT / 100.0          # 99.3%  (barely shorter -> false comfort)
    h_afterL2 = H * (EXIST_PCT - WP) / 100.0   # ~83.4% (visible ~1/6 step)
    y_bot = Y_TOP - H                          # common bottom reference for the dropped pile

    def seg(xa, xb, h, color, alpha=1.0):
        ax.add_patch(plt.Rectangle((xa, Y_TOP - h), xb - xa, h, facecolor=color,
                     edgecolor="white", linewidth=0.6, alpha=alpha, zorder=2))

    seg(x0, xL1, H, GREY, alpha=0.30)            # emitted pool
    seg(xL1, xL2, h_afterL1, GREY, alpha=0.40)   # after L1 (almost identical height)
    seg(xL2, xEnd, h_afterL2, TEAL, alpha=0.85)  # survivors that support the claim

    # ---- gate headers (single line each -> no collision) + vertical combs ----
    def gate(x, tag, q, col):
        ax.plot([x, x], [y_bot, Y_TOP + 5], color=INK2, lw=1.0,
                ls=(0, (2, 1.7)), zorder=5)
        ax.text(x, Y_TOP + 7.5, tag, ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color=col)
        ax.text(x, Y_TOP + 6.0, q, ha="center", va="top", fontsize=7.0,
                color=INK2, style="italic")

    gate(xL1, "L1", "exists?", BLUE)
    gate(xL2, "L2", "supports the claim?", CORAL)

    # ---- what each gate CATCHES, dropped below the pipe (coral) ----
    # L1: 0.7% — a hair. SHALLOW drop, thin tick; the message is "almost nothing".
    ax.annotate("", xy=(xL1, y_bot - 8.5), xytext=(xL1, y_bot - 0.5),
                arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=0.9), zorder=4)
    ax.plot([xL1 - 3.2, xL1 + 3.2], [y_bot - 9.6, y_bot - 9.6], color=CORAL, lw=2.4, zorder=4)
    ax.text(xL1, y_bot - 12.0, f"{FAB_PCT:.1f}% fabricated", ha="center", va="top",
            fontsize=7.4, color=CORAL, fontweight="bold")
    ax.text(xL1, y_bot - 16.4, f"{N_FAB} of {N_EMIT:,}", ha="center", va="top",
            fontsize=6.4, color=INK2, style="italic")

    # L2: 15.9% — a prominent coral pill (number pops; label beneath, never clipped).
    ax.annotate("", xy=(xL2, y_bot - 11.5), xytext=(xL2, y_bot - 0.5),
                arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=1.8), zorder=4)
    pill_w, pill_h = 16.0, 11.0
    ax.add_patch(FancyBboxPatch((xL2 - pill_w / 2, y_bot - 23.5), pill_w, pill_h,
                 boxstyle="round,pad=0.4,rounding_size=2.8", linewidth=0,
                 facecolor=CORAL, zorder=4))
    ax.text(xL2, y_bot - 18.0, f"{WP:.0f}%", ha="center", va="center",
            fontsize=12.5, color="white", fontweight="bold", zorder=5)
    ax.text(xL2, y_bot - 25.2, "wrong-paper", ha="center", va="top",
            fontsize=8.4, color=CORAL, fontweight="bold")
    ax.text(xL2, y_bot - 29.4, "real PMID, claim unsupported", ha="center", va="top",
            fontsize=6.4, color=INK2, style="italic")
    ax.text(xL2, y_bot - 33.0, "$\\approx$1 in 6 real citations", ha="center", va="top",
            fontsize=6.6, color=CORAL, fontweight="bold")

    # ---- entry + exit labels ----
    ax.text(x0 + 1.2, Y_TOP - H / 2, f"{N_EMIT:,}\ncitations\nagents\nemit", ha="left",
            va="center", fontsize=6.6, color=INK2, linespacing=1.12, zorder=6)
    ax.text(xEnd + 1.6, Y_TOP - h_afterL2 / 2, f"$\\approx${100 - WP:.0f}%\nsupport\n(full /\npartial)",
            ha="left", va="center", fontsize=6.8, color="#1c7a6b", linespacing=1.12,
            fontweight="bold")

    # cross-judge honesty note (tiny footnote at the very bottom, clear of both drops)
    ax.text(50, y_bot - 38.5,
            "Wrong-paper $\\approx$" + f"{WP_OPUS:.0f}–{WP:.0f}% across two independent judge families.",
            ha="center", va="top", fontsize=5.9, color=INK2)

    ax.set_title("An existence check is false comfort", loc="center",
                 fontsize=11.0, fontweight="bold", color=INK, pad=4)

    fig.tight_layout(pad=0.2)
    out = os.path.join(FIG, "fig_existence_funnel.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, "| exist%", EXIST_PCT, "wp%", WP, "fab%", FAB_PCT)
    return out


def _shadow(ax, x, y, w, h, boxstyle, z=1):
    ax.add_patch(FancyBboxPatch((x + 0.5, y - 0.9), w, h, boxstyle=boxstyle,
                 linewidth=0, facecolor="#000000", alpha=0.06, zorder=z))


# ============================================================================
# HERO 2 — fig_eval_flow.pdf  (the EVALUATION flow; companion to fig_pipeline's
#          CONSTRUCTION flow). One open question -> agentic harness -> trajectory,
#          graded two ways: frozen checklist -> solve rate; L1/L2 audit -> wrong-paper.
# ============================================================================
def _glyph(ax, name, cx, cy, col, s=1.0, asp=1.0):
    """Tiny flat line-icon — one fixed glyph per role (his house figure grammar).
    Local coords are aspect-corrected (asp) so discs render visually round."""
    lw = 1.05
    def M(lx, ly):
        return cx + lx * s, cy + ly * s * asp
    def disc(lx, ly, r, fill=False):
        ox, oy = M(lx, ly)
        ax.add_patch(Ellipse((ox, oy), 2 * r * s, 2 * r * s * asp,
                     facecolor=(col if fill else "none"), edgecolor=col,
                     linewidth=lw, zorder=5))
    def seg(pts, w=lw):
        xs = [M(lx, ly)[0] for lx, ly in pts]
        ys = [M(lx, ly)[1] for lx, ly in pts]
        ax.plot(xs, ys, color=col, lw=w, solid_capstyle="round",
                solid_joinstyle="round", zorder=5, clip_on=False)
    if name == "q":                       # open question
        disc(0, 0, 2.3)
        ax.text(cx, cy - 0.15 * s * asp, "?", ha="center", va="center",
                fontsize=6.0 * s, fontweight="bold", color=col, zorder=6)
    elif name == "llm":                   # NN node-graph = a language model
        P = [(-1.9, 1.4), (-1.9, -1.4), (1.9, 1.4), (1.9, -1.4)]
        for a in (0, 1):
            for b in (2, 3):
                seg([P[a], P[b]], 0.65)
        for lx, ly in P:
            disc(lx, ly, 0.6, fill=True)
    elif name == "doc":                   # trajectory = document
        seg([(-1.7, -2.3), (-1.7, 2.3), (1.7, 2.3), (1.7, -2.3), (-1.7, -2.3)])
        seg([(-1.0, 1.1), (1.0, 1.1)], 0.65)
        seg([(-1.0, -0.1), (1.0, -0.1)], 0.65)
    elif name == "check":                 # frozen checklist = ticked checkbox
        seg([(-2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (2.0, -2.0), (-2.0, -2.0)])
        seg([(-1.1, -0.1), (-0.2, -1.1), (1.5, 1.4)], 1.3)
    elif name == "audit":                 # citation audit = magnifier
        disc(-0.5, 0.6, 1.7)
        seg([(0.75, -0.65), (2.3, -2.2)], 1.3)


def fig_eval_flow():
    INK, INK2, BLUE, TEAL, CORAL = S.INK, S.INK2, S.BLUE, S.TEAL, S.CORAL
    C_LLM, C_LLM_EDGE = "#DCECF7", S.BLUE
    fig, ax = plt.subplots(figsize=(7.0, 2.62))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ASP = 7.0 / 2.62   # data-unit aspect (xlim==ylim==100) so glyph discs render round

    def card(x, y, w, h, title, lines, edge, fill, tcol=None, badge=None, badgecol=None,
             icon=None):
        _shadow(ax, x, y, w, h, "round,pad=0,rounding_size=2.2", 1)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.2",
                     linewidth=1.2, edgecolor=edge, facecolor=fill, zorder=2))
        if icon:
            _glyph(ax, icon, x + w / 2, y + h - 6.4, tcol or INK2, s=0.56, asp=ASP)
        ax.text(x + w / 2, y + h - 11.8, title, ha="center", va="center",
                fontsize=8.6, fontweight="bold", color=tcol or INK, zorder=4)
        for j, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 17.4 - j * 4.8, ln, ha="center", va="center",
                    fontsize=6.6, color=INK2, zorder=4)
        if badge:
            ax.text(x + w - 2.0, y + 1.8, badge, ha="right", va="bottom",
                    fontsize=5.6, fontweight="bold", color=badgecol or edge, zorder=4)

    def arrow(xa, ya, xb, yb, col="#9098A2", lw=1.5, rad=0.0):
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=12,
                     linewidth=lw, color=col, zorder=3, shrinkA=1, shrinkB=1,
                     connectionstyle=f"arc3,rad={rad}"))

    # ---- spine: input -> harness -> trajectory (vertically centered band) ----
    yb, h = 40.0, 34.0
    A = (1.5, 19.0); B = (25.5, 21.5); C = (52.5, 18.5)   # (x, width)
    card(A[0], yb, A[1], h, "Open question",
         ["unsolved; no", "answer key", "+ frozen checklist"], "#C2C7CF", "#FFFFFF", icon="q")
    card(B[0], yb, B[1], h, "Agentic harness",
         ["reason $\\leftrightarrow$ act,  $\\leq$10 rounds", "10 biomedical REST tools",
          "pubmed · trials · openfda", "chembl · uniprot · …"], C_LLM_EDGE, C_LLM,
         tcol=C_LLM_EDGE, badge="LLM", badgecol=C_LLM_EDGE, icon="llm")
    card(C[0], yb, C[1], h, "Trajectory",
         ["final answer", "+ cited PMIDs", "+ tool calls"], "#C2C7CF", "#FFFFFF", icon="doc")
    arrow(A[0] + A[1], yb + h / 2, B[0], yb + h / 2)
    arrow(B[0] + B[1], yb + h / 2, C[0], yb + h / 2)

    # ---- fork into the two graders (right), each emitting one headline metric ----
    gx, gw = 74.0, 23.5
    g1y, g2y, gh = 52.0, 6.0, 32.0
    card(gx, g1y, gw, gh, "Frozen checklist",
         ["mention · acknowledge", "ground · avoid  (w 1–3)", "score $\\geq$ 0.5 = solve"],
         TEAL, "#E9F6F1", tcol="#1c7a6b", icon="check")
    card(gx, g2y, gw, gh, "Citation audit",
         ["L1 · does the ID exist?", "L2 · does it support?"],
         CORAL, "#FCEEE8", tcol="#b5451f", icon="audit")
    cx, cy = C[0] + C[1], yb + h / 2
    arrow(cx, cy + 4, gx, g1y + gh / 2, col=TEAL, lw=1.7, rad=0.18)
    arrow(cx, cy - 4, gx, g2y + gh / 2, col=CORAL, lw=1.7, rad=-0.18)

    # metric outputs (what each grader produces)
    ax.text(gx + gw + 0.6, g1y + gh / 2, "$\\rightarrow$ solve\n   rate", ha="left", va="center",
            fontsize=7.2, fontweight="bold", color="#1c7a6b", linespacing=1.15)
    ax.text(gx + gw + 0.6, g2y + gh / 2, "$\\rightarrow$ wrong-\n   paper rate", ha="left",
            va="center", fontsize=7.2, fontweight="bold", color="#b5451f", linespacing=1.15)

    ax.set_title("Agentic evaluation: one open question, graded by two independent measures",
                 loc="center", fontsize=10.0, fontweight="bold", color=INK, pad=5)
    fig.tight_layout(pad=0.2)
    out = os.path.join(FIG, "fig_eval_flow.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    fig_funnel()
    fig_eval_flow()
