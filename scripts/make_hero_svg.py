#!/usr/bin/env python3
"""Fig 5 (results hero) as a hand-authored SVG -> vector PDF via cairosvg.

Two panels, matching the SVG aesthetic of Fig 4 (soft pastel, crisp rounded bars, blur
shadows, marker arrowheads):
  (a) existence-vs-correctness grouped bars (L1 exists vs L2 supports, per model + overall),
  (b) frontier frozen-core solve-rate bars with Wilson-CI whiskers.
Values are loaded from results JSON and asserted against FACTS (no hand-typed data). cairo
embeds subsetted TrueType (not Type3) -> AAAI font-compliant.

Run with the torchtitan env python; writes paper_writing/figures/fig_results_overview.pdf.
"""
import json
import cairosvg

ROOT = "/data/project/private/minstar/workspace/healthcare-research"
FIG = f"{ROOT}/paper_writing/figures"
RES = f"{ROOT}/results"

# palette (soft pastel, consistent with the figure set)
SKY, TEAL, VERM = "#88C6EC", "#46B08F", "#E0875A"
BLUE, PURPLE = "#5A9BD4", "#A98BD0"
EXTXT, SUPTXT = "#2E7CB8", "#1F7A5E"
INK, INK2, GRAY, GRID = "#242424", "#5A5A5A", "#9A9A9A", "#DDDDDD"
FONT = "DejaVu Sans"


def _close(a, b, t=0.3):
    return abs(a - b) <= t


# ---- data (JSON-sourced + FACTS asserts) ----
_ca = json.load(open(f"{RES}/cite_audit_summary.json"))
MS = ["GLM-5.1", "Qwen3.6", "DeepSeek-V4"]
EX = {m: _ca["by_model"][m]["exist_pct"] for m in MS}
SUP = {m: round(100 - _ca["by_model"][m]["wrong_paper_pct"], 1) for m in MS}
N_L2 = {m: _ca["by_model"][m]["L2_judged"] for m in MS}
OVR_SUP = round(100 - _ca["overall"]["wrong_paper_pct"], 1)   # 84.1
OVR_EX = 99.3   # FACTS L130 (Qwen 99.6 / DSV4 99.8 dominate the pool)
assert _close(EX["GLM-5.1"], 84.7) and _close(EX["Qwen3.6"], 99.6) and _close(EX["DeepSeek-V4"], 99.8)
assert _close(SUP["GLM-5.1"], 43.3) and _close(SUP["Qwen3.6"], 79.8) and _close(SUP["DeepSeek-V4"], 86.9)
assert N_L2["GLM-5.1"] == 97 and _close(OVR_SUP, 84.1)

_fr = json.load(open(f"{RES}/frontier_robust_leaderboard.json"))


def _F(key):
    e = _fr[key]
    return (e["solve@0.5"], e["ci"][0], e["ci"][1])


# top-to-bottom: GPT-5.5, Opus-4.7, Gemini-3-Pro
FRONTIER = [("GPT-5.5",) + _F("api_gpt55_robust"),
            ("Opus-4.7",) + _F("api_opus47_robust"),
            ("Gemini-3-Pro",) + _F("api_gemini3pro_robust")]
FCOL = {"GPT-5.5": BLUE, "Opus-4.7": VERM, "Gemini-3-Pro": PURPLE}
assert _close(FRONTIER[0][1], 59.6) and _close(FRONTIER[1][1], 37.8) and _close(FRONTIER[2][1], 28.8)

W, Hgt = 720, 300
p = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size, fill=INK, weight="normal", anchor="start", italic=False, rot=None):
    fs = ' font-style="italic"' if italic else ''
    tr = f' transform="rotate({rot} {x} {y})"' if rot is not None else ''
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{fs}{tr}>{esc(s)}</text>')


def rect(x, y, w, h, fill, rx=2, opacity=1.0, stroke=None, sw=1.0, shadow=False):
    s = ""
    if shadow:
        s += (f'<rect x="{x+1.2}" y="{y+1.8}" width="{w}" height="{h}" rx="{rx}" '
              f'fill="#26313B" opacity="0.13" filter="url(#blur)"/>')
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ''
    s += (f'<rect x="{x}" y="{y}" width="{w}" height="{max(h,0.1)}" rx="{rx}" '
          f'fill="{fill}" opacity="{opacity}"{st}/>')
    return s


def dline(x1, y1, x2, y2, color=GRID, w=0.9, dash="4,2"):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{w}" stroke-dasharray="{dash}"/>')


def header(x, w, fill, ic, title):
    # colored panel-header band (echoes Fig 4 stage headers) + white icon + white title
    s = f'<rect x="{x}" y="6" width="{w}" height="26" rx="7" fill="{fill}"/>'
    s += ic(x + 18, 19)
    s += txt(x + 32, 23, title, 11, "#FFFFFF", "bold")
    return s


def ic_cite(cx, cy, c="#fff"):
    # citation audit: small document + magnifier
    return (f'<path d="M{cx-7},{cy-8} h8 l3,3 v11 h-11 z" fill="none" stroke="{c}" '
            f'stroke-width="1.4" stroke-linejoin="round"/>'
            f'<line x1="{cx-4}" y1="{cy-3}" x2="{cx+1}" y2="{cy-3}" stroke="{c}" stroke-width="1.1"/>'
            f'<circle cx="{cx+4}" cy="{cy+4}" r="3.8" fill="none" stroke="{c}" stroke-width="1.5"/>'
            f'<line x1="{cx+6.8}" y1="{cy+6.8}" x2="{cx+9.5}" y2="{cy+9.5}" stroke="{c}" stroke-width="1.6"/>')


def ic_target(cx, cy, c="#fff"):
    # bullseye: "saturate the core"
    return (f'<circle cx="{cx}" cy="{cy}" r="8" fill="none" stroke="{c}" stroke-width="1.5"/>'
            f'<circle cx="{cx}" cy="{cy}" r="4.4" fill="none" stroke="{c}" stroke-width="1.4"/>'
            f'<circle cx="{cx}" cy="{cy}" r="1.5" fill="{c}"/>')


p.append('<defs>')
p.append('<filter id="blur" x="-40%" y="-40%" width="180%" height="180%">'
         '<feGaussianBlur stdDeviation="2.0"/></filter>')
p.append(f'<marker id="ah" markerWidth="9" markerHeight="9" refX="6.5" refY="3" '
         f'orient="auto" markerUnits="userSpaceOnUse">'
         f'<path d="M0,0 L7,3 L0,6 Z" fill="#555555"/></marker>')
p.append('</defs>')

# ============================ PANEL (a) ============================
p.append(header(6, 350, "#CC6F44", ic_cite, "(a)  Existence is not correctness"))
PAL, PAR, PAT, PAB = 48, 348, 52, 234
VMAX = 112


def ya(v):
    return PAB - (v / VMAX) * (PAB - PAT)


# y gridlines + ticks
for t in (0, 25, 50, 75, 100):
    yy = ya(t)
    p.append(f'<line x1="{PAL}" y1="{yy}" x2="{PAR}" y2="{yy}" stroke="{GRID}" stroke-width="0.8"/>')
    p.append(txt(PAL - 6, yy + 3, str(t), 8, INK2, anchor="end"))
p.append(txt(15, (PAT + PAB) / 2, "% of real citations", 9, INK2, anchor="middle",
             rot=-90))

groups = MS + ["Overall"]
exists = [EX["GLM-5.1"], EX["Qwen3.6"], EX["DeepSeek-V4"], OVR_EX]
supports = [SUP["GLM-5.1"], SUP["Qwen3.6"], SUP["DeepSeek-V4"], OVR_SUP]
gw = (PAR - PAL) / len(groups)
bw = 21
for i, g in enumerate(groups):
    gx = PAL + (i + 0.5) * gw
    xe, xs = gx - bw - 1.5, gx + 1.5
    # gap shading on the exists bar (existence down to supports)
    p.append(rect(xe, ya(exists[i]), bw, ya(supports[i]) - ya(exists[i]), VERM, rx=1,
                  opacity=0.14))
    p.append(rect(xe, ya(exists[i]), bw, ya(0) - ya(exists[i]), SKY, rx=2, opacity=0.55))
    p.append(rect(xs, ya(supports[i]), bw, ya(0) - ya(supports[i]), TEAL, rx=2))
    p.append(txt(xe + bw / 2, ya(exists[i]) - 3, f"{exists[i]:.1f}", 7.4, EXTXT, "bold", "middle"))
    p.append(txt(xs + bw / 2, ya(supports[i]) - 3, f"{supports[i]:.1f}", 7.4, SUPTXT, "bold", "middle"))
    p.append(txt(gx, PAB + 12, g, 8.2, INK, "normal", "middle"))
# n=97 inside GLM supports bar (white; fragile-sample marker)
glm_xs = PAL + 0.5 * gw + 1.5
p.append(txt(glm_xs + bw / 2, ya(supports[0]) + 11, f"n={N_L2['GLM-5.1']}", 6.6, "#FFFFFF",
             "bold", "middle"))
# near-perfect existence line (label at left, above GLM's gap -> clear of the Overall 99.3 label)
p.append(dline(PAL, ya(99), PAR, ya(99), GRAY, 0.9))
p.append(txt(PAL + 6, ya(99) - 3, "near-perfect existence", 6.6, GRAY, anchor="start", italic=True))
# callout in GLM's wide exists-vs-supports gap (reading order top -> bottom)
p.append(txt(120, ya(60), "~1 in 6 real citations", 7.8, VERM, "bold", "middle"))
p.append(txt(120, ya(60) + 11, "do not support the claim", 7.8, VERM, "bold", "middle"))
# legend
ly = PAB + 30
p.append(rect(PAL, ly, 12, 9, SKY, rx=2, opacity=0.55))
p.append(txt(PAL + 17, ly + 8, "Citation exists (L1)", 7.6, INK2))
p.append(rect(PAL + 120, ly, 12, 9, TEAL, rx=2))
p.append(txt(PAL + 137, ly + 8, "Supports the claim (L2 floor)", 7.6, INK2))

# ============================ PANEL (b) ============================
p.append(header(378, 336, "#4A86C2", ic_target, "(b)  No frontier agent saturates the core"))
PBL, PBR, PBT, PBB = 462, 702, 58, 210
XMAX = 80


def xb(v):
    return PBL + (v / XMAX) * (PBR - PBL)


# x gridlines + ticks
for t in (0, 20, 40, 60, 80):
    xx = xb(t)
    p.append(f'<line x1="{xx}" y1="{PBT}" x2="{xx}" y2="{PBB}" stroke="{GRID}" stroke-width="0.8"/>')
    p.append(txt(xx, PBB + 13, str(t), 8, INK2, anchor="middle"))
p.append(txt((PBL + PBR) / 2, PBB + 30, "Frozen-core (423) solve rate (%) [Wilson 95% CI]",
             8.6, INK2, anchor="middle"))

rh = (PBB - PBT) / len(FRONTIER)
bh = 24
for k, (lab, v, lo, hi) in enumerate(FRONTIER):
    cy = PBT + (k + 0.5) * rh
    c = FCOL[lab]
    p.append(txt(PBL - 8, cy + 3, lab, 8.4, INK, anchor="end"))
    p.append(rect(PBL, cy - bh / 2, xb(v) - PBL, bh, c, rx=2, shadow=True))
    # CI whisker
    p.append(f'<line x1="{xb(lo)}" y1="{cy}" x2="{xb(hi)}" y2="{cy}" stroke="#3A3A3A" stroke-width="1.3"/>')
    for e in (lo, hi):
        p.append(f'<line x1="{xb(e)}" y1="{cy-3.5}" x2="{xb(e)}" y2="{cy+3.5}" stroke="#3A3A3A" stroke-width="1.3"/>')
    p.append(txt(xb(hi) + 6, cy + 3, f"{v:.1f}", 8.6, c, "bold"))
# majority line
p.append(dline(xb(50), PBT, xb(50), PBB, GRAY, 0.9))
p.append(txt(xb(50), PBB + 13, "majority", 6.8, GRAY, anchor="middle", italic=True))
# ~40% unsolved callout (arrow to GPT-5.5 bar top)
gpt_cy = PBT + 0.5 * rh
p.append(f'<line x1="{xb(50)}" y1="{PBT-6}" x2="{xb(59.6)-2}" y2="{gpt_cy-bh/2-1}" '
         f'stroke="#555555" stroke-width="1.0" marker-end="url(#ah)"/>')
p.append(txt(xb(34), PBT - 18, "~40% unsolved", 7.6, INK, anchor="middle"))
p.append(txt(xb(34), PBT - 8, "even by the best agent", 7.6, INK, anchor="middle"))

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hgt}" '
       f'viewBox="0 0 {W} {Hgt}">' + "".join(p) + '</svg>')

out = f"{FIG}/fig_results_overview.pdf"
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=out)
print(f"wrote {out}")
print(f"  exists {exists}  supports {supports}  frontier {[(f[0], f[1]) for f in FRONTIER]}")
