#!/usr/bin/env python3
"""Fig 4 (data-construction pipeline) as a hand-authored SVG -> vector PDF via cairosvg.

Illustrator/BioRender-style clean schematic; crisper and more "designed" than matplotlib
box-drawing for a framework figure (real blur drop-shadows, marker arrowheads, precise
rounded cards). Numbers are FACTS-only. The integrated example is a *real* corpus item
(PMID 40578802) showing the self-contained rewrite. cairo embeds subsetted TrueType (not
Type3), so the output is AAAI font-compliant.

Run with the torchtitan env python; writes paper_writing/figures/fig_pipeline.pdf.
"""
import cairosvg

ROOT = "/data/project/private/minstar/workspace/healthcare-research"
FIG = f"{ROOT}/paper_writing/figures"

# soft pastel palette (matches the rest of the figure set)
BLUE, TEAL, GOLD, VERM = "#5A9BD4", "#46B08F", "#E8B85E", "#E0875A"
LLM_FILL, CARD, CARD_EDGE, BAND = "#DCECF7", "#FFFFFF", "#C7CCD4", "#F5F8FB"
INK, INK2, GRAY, LABEL = "#242424", "#5A5A5A", "#868C95", "#5A6473"
FONT = "DejaVu Sans"

# FACTS-only numbers
SC_BEFORE, SC_AFTER = 51.6, 85.4
N_BASE, N_DOMAINS, N_CORE, N_ROBUST = "12,553", 12, 657, 423
FLIP_PCT, UNK_PCT = 56.5, 14

W, Hgt = 720, 350
p = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rrect(x, y, w, h, rx, fill, stroke=None, sw=1.4, shadow=False):
    s = ""
    if shadow:
        s += (f'<rect x="{x+1.6}" y="{y+2.7}" width="{w}" height="{h}" rx="{rx}" '
              f'fill="#26313B" opacity="0.15" filter="url(#blur)"/>')
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ''
    s += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{st}/>'
    return s


def txt(x, y, s, size, fill=INK, weight="normal", anchor="start", italic=False):
    fs = ' font-style="italic"' if italic else ''
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{fs}>{esc(s)}</text>')


def lines(x, y, rows, size, fill, lh, anchor="start", italic=False):
    return "".join(txt(x, y + i * lh, r, size, fill, anchor=anchor, italic=italic)
                   for i, r in enumerate(rows))


# ---- defs: blur filter + arrowhead markers ----
p.append('<defs>')
p.append('<filter id="blur" x="-40%" y="-40%" width="180%" height="180%">'
         '<feGaussianBlur stdDeviation="2.3"/></filter>')
for mid, col in [("ahg", "#9098A2"), ("ahv", VERM)]:
    p.append(f'<marker id="{mid}" markerWidth="9" markerHeight="9" refX="6.5" refY="3" '
             f'orient="auto" markerUnits="userSpaceOnUse">'
             f'<path d="M0,0 L7,3 L0,6 Z" fill="{col}"/></marker>')
p.append('</defs>')

# ---- CONSTRUCTION band ----
BX, BY, BW, BH = 6, 6, W - 12, 150
p.append(rrect(BX, BY, BW, BH, 12, BAND))
p.append(txt(20, 27, "CONSTRUCTION", 12.5, LABEL, "bold"))

# ---- 5 stage cards ----
stages = [
    ("1", "Crawl", False, ["PubMed · trials · arXiv", "JLA · NICE · WHO/CHNRI",
                           "Cochrane research gaps"]),
    ("2", "Extract", True, ["≥ 1 question / paper", "unique task id"]),
    ("3", "Refine", True, ["+ taxonomy, tools,", "difficulty hints",
                           "+ self-contained rewrite"]),
    ("4", "Dedup", False, ["near-duplicate removal", "MiniLM-L6 cos ≥ 0.90"]),
    ("5", "Export", False, [f"{N_BASE} questions", f"{N_DOMAINS} domains"]),
]
CY, CH = 42, 100
left, right = 20, W - 20
gap = 13
cw = (right - left - gap * (len(stages) - 1)) / len(stages)
centers = []
for i, (num, title, llm, body) in enumerate(stages):
    x = left + i * (cw + gap)
    cx = x + cw / 2
    centers.append((x, cx))
    fill = LLM_FILL if llm else CARD
    edge = BLUE if llm else CARD_EDGE
    acc = BLUE if llm else GRAY
    p.append(rrect(x, CY, cw, CH, 9, fill, edge, 1.5, shadow=True))
    p.append(f'<circle cx="{x+16}" cy="{CY+17}" r="9" fill="{acc}"/>')
    p.append(txt(x + 16, CY + 20.5, num, 9.5, "#FFFFFF", "bold", "middle"))
    p.append(txt(x + 30, CY + 21, title, 13, INK, "bold"))
    if llm:
        p.append(txt(x + cw - 7, CY + CH - 8, "LLM", 8, BLUE, "bold", "end"))
    n = len(body)
    y0 = CY + 44 + (3 - n) * 6
    p.append(lines(cx, y0, body, 9.2, INK2, 13.5, anchor="middle"))

# ---- flow arrows between cards ----
ay = CY + CH / 2
for i in range(len(stages) - 1):
    x_end = centers[i][0] + cw
    x_nxt = centers[i + 1][0]
    p.append(f'<line x1="{x_end+1.5}" y1="{ay}" x2="{x_nxt-2.5}" y2="{ay}" '
             f'stroke="#9098A2" stroke-width="2.1" marker-end="url(#ahg)"/>')

# ---- integrated example: what Refine does (real corpus item) ----
p.append(txt(20, 178, "What “Refine” does — self-contained rewrite "
             "(real corpus item, PMID 40578802):", 10, GRAY, "normal", italic=True))
EBY, EBH = 186, 66
# original (extracted)
p.append(rrect(20, EBY, 268, EBH, 8, CARD, CARD_EDGE, 1.3, shadow=True))
p.append(txt(32, EBY + 18, "original (extracted)", 8.8, GRAY, "bold"))
p.append(lines(32, EBY + 36, ["“The pathophysiology of PSVD",
                              "remains unclear”"], 10, INK, 15, italic=True))
# self-contained question
p.append(rrect(432, EBY, 268, EBH, 8, CARD, TEAL, 1.5, shadow=True))
p.append(txt(444, EBY + 18, "self-contained question", 8.8, TEAL, "bold"))
p.append(lines(444, EBY + 35, ["“What is the pathophysiologic",
                               "mechanism of portosinusoidal",
                               "vascular disorder (PSVD)?”"], 9.6, INK, 13.5, italic=True))
# arrow + metric
ax_y = EBY + EBH / 2
p.append(f'<line x1="290" y1="{ax_y}" x2="430" y2="{ax_y}" stroke="{VERM}" '
         f'stroke-width="2.4" marker-end="url(#ahv)"/>')
p.append(txt(360, ax_y - 19, "self-containment", 9, VERM, "bold", "middle"))
p.append(txt(360, ax_y - 8, f"{SC_BEFORE}% → {SC_AFTER}%", 9, VERM, "bold", "middle"))

# ---- two additive tracks ----
p.append(txt(20, 268, "then layered on additively:", 9.5, GRAY, "normal", italic=True))
TBY, TBH = 274, 70
# (a) openness
p.append(rrect(20, TBY, 333, TBH, 8, "#EAF3FA", BLUE, 1.4, shadow=True))
p.append(txt(34, TBY + 19, "(a)  Retrieval-grounded openness", 11, BLUE, "bold"))
p.append(lines(34, TBY + 33, [
    "status verifier → Stage-2 judge",
    "source-framing: 0 answered/unknown",
    f"grounded: {FLIP_PCT}% labels flip, {UNK_PCT}% unknown",
    f"0/{N_CORE} core questions still resolved"], 8.8, INK, 11))
# (b) difficulty
p.append(rrect(367, TBY, 333, TBH, 8, "#E9F6F1", TEAL, 1.4, shadow=True))
p.append(txt(381, TBY + 19, "(b)  Empirical difficulty", 11, TEAL, "bold"))
p.append(lines(381, TBY + 33, [
    "GLM-5.1, Qwen3.6, DeepSeek-V4",
    "answer + grade every question",
    f"all three fail → core ({N_CORE})",
    f"→ frozen core ({N_ROBUST} at T=0)"], 8.8, INK, 11))

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hgt}" '
       f'viewBox="0 0 {W} {Hgt}">' + "".join(p) + '</svg>')

out = f"{FIG}/fig_pipeline.pdf"
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=out)
print(f"wrote {out}")
