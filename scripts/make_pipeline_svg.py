#!/usr/bin/env python3
"""Fig 4 (data-construction pipeline) as a hand-authored SVG -> vector PDF via cairosvg.

Infographic style (design ref: Gemini mockups in paper_writing/figure_ref): per-stage colored
header bands + flat line icons + soft body cards + an integrated REAL corpus example
(PMID 40578802) for the self-contained rewrite. Content/numbers are FACTS-only (the Gemini
mockups' labels are AI artifacts and are NOT used). cairo embeds subsetted TrueType (not Type3),
so the output is AAAI font-compliant.

Run with the torchtitan env python; writes paper_writing/figures/fig_pipeline.pdf.
"""
import cairosvg

ROOT = "/data/project/private/minstar/workspace/healthcare-research"
FIG = f"{ROOT}/paper_writing/figures"

# soft body palette + saturated header bands (white title/icon reads on the headers)
BLUE, TEAL, GOLD, VERM = "#5A9BD4", "#46B08F", "#E8B85E", "#E0875A"
HDR = ["#4A86C2", "#7A6FC0", "#A85FB0", "#C75397", "#D14E82"]  # blue -> magenta gradient
CARD, BAND = "#FFFFFF", "#F5F8FB"
INK, INK2, GRAY, LABEL = "#242424", "#5A5A5A", "#868C95", "#5A6473"
FONT = "DejaVu Sans"

SC_BEFORE, SC_AFTER = 51.6, 85.4
N_BASE, N_DOMAINS, N_CORE, N_ROBUST = "12,553", 12, 657, 423
FLIP_PCT, UNK_PCT = 56.5, 14

W, Hgt = 720, 350
p = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size, fill=INK, weight="normal", anchor="start", italic=False):
    fs = ' font-style="italic"' if italic else ''
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{fs}>{esc(s)}</text>')


def lines(x, y, rows, size, fill, lh, anchor="start", italic=False):
    return "".join(txt(x, y + i * lh, r, size, fill, anchor=anchor, italic=italic)
                   for i, r in enumerate(rows))


def rrect(x, y, w, h, rx, fill, stroke=None, sw=1.4, shadow=False):
    s = ""
    if shadow:
        s += (f'<rect x="{x+1.6}" y="{y+2.7}" width="{w}" height="{h}" rx="{rx}" '
              f'fill="#26313B" opacity="0.15" filter="url(#blur)"/>')
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ''
    s += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{st}/>'
    return s


def top_round(x, y, w, h, r, fill):
    # rect with only the TOP two corners rounded (radius r)
    return (f'<path d="M{x},{y+r} a{r},{r} 0 0 1 {r},-{r} h{w-2*r} '
            f'a{r},{r} 0 0 1 {r},{r} v{h-r} h{-w} z" fill="{fill}"/>')


# ---- flat line icons (white on colored header), centered at (cx, cy) ----
def ic_db(cx, cy, c="#fff"):
    return (f'<ellipse cx="{cx}" cy="{cy-6}" rx="8" ry="3" fill="none" stroke="{c}" stroke-width="1.6"/>'
            f'<path d="M{cx-8},{cy-6} V{cy+5} a8,3 0 0 0 16,0 V{cy-6}" fill="none" stroke="{c}" stroke-width="1.6"/>'
            f'<path d="M{cx-8},{cy-0.5} a8,3 0 0 0 16,0" fill="none" stroke="{c}" stroke-width="1.3"/>')


def ic_doc(cx, cy, c="#fff"):
    return (f'<path d="M{cx-6},{cy-8} h9 l4,4 v12 h-13 z" fill="none" stroke="{c}" '
            f'stroke-width="1.5" stroke-linejoin="round"/>'
            f'<line x1="{cx-3}" y1="{cy-1}" x2="{cx+4}" y2="{cy-1}" stroke="{c}" stroke-width="1.2"/>'
            f'<line x1="{cx-3}" y1="{cy+3}" x2="{cx+4}" y2="{cy+3}" stroke="{c}" stroke-width="1.2"/>')


def ic_spark(cx, cy, c="#fff"):
    return (f'<path d="M{cx},{cy-9} L{cx+2.6},{cy-2.6} L{cx+9},{cy} L{cx+2.6},{cy+2.6} '
            f'L{cx},{cy+9} L{cx-2.6},{cy+2.6} L{cx-9},{cy} L{cx-2.6},{cy-2.6} Z" fill="{c}"/>')


def ic_dedup(cx, cy, c="#fff"):
    return (f'<circle cx="{cx-3.5}" cy="{cy}" r="6" fill="none" stroke="{c}" stroke-width="1.5"/>'
            f'<circle cx="{cx+3.5}" cy="{cy}" r="6" fill="none" stroke="{c}" stroke-width="1.5"/>')


def ic_chart(cx, cy, c="#fff"):
    return (f'<rect x="{cx-7.5}" y="{cy+1}" width="3.6" height="7" rx="1" fill="{c}"/>'
            f'<rect x="{cx-1.8}" y="{cy-3}" width="3.6" height="11" rx="1" fill="{c}"/>'
            f'<rect x="{cx+3.9}" y="{cy-7}" width="3.6" height="15" rx="1" fill="{c}"/>')


ICONS = [ic_db, ic_doc, ic_spark, ic_dedup, ic_chart]

# ---- defs ----
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

# ---- 5 stage cards (colored header + icon + soft body) ----
stages = [
    ("Crawl", False, ["PubMed · trials · arXiv", "JLA · NICE · WHO/CHNRI",
                      "Cochrane research gaps"]),
    ("Extract", True, ["≥ 1 question / paper", "unique task id"]),
    ("Refine", True, ["+ taxonomy, tools,", "difficulty hints",
                      "+ self-contained rewrite"]),
    ("Dedup", False, ["near-duplicate removal", "MiniLM-L6 cos ≥ 0.90"]),
    ("Export", False, [f"{N_BASE} questions", f"{N_DOMAINS} domains"]),
]
CY, CH, HH = 42, 100, 28
left, right, gap = 20, W - 20, 13
cw = (right - left - gap * (len(stages) - 1)) / len(stages)
centers = []
for i, (title, llm, body) in enumerate(stages):
    x = left + i * (cw + gap)
    cx = x + cw / 2
    centers.append((x, cx))
    p.append(rrect(x, CY, cw, CH, 9, CARD, "#D7DCE3", 1.0, shadow=True))
    p.append(top_round(x, CY, cw, HH, 9, HDR[i]))
    p.append(ICONS[i](x + 16, CY + HH / 2))
    p.append(txt(x + 29, CY + HH / 2 + 4, f"{i+1}  {title}", 11.5, "#FFFFFF", "bold"))
    if llm:
        p.append(txt(x + cw - 7, CY + CH - 8, "LLM", 8, HDR[i], "bold", "end"))
    n = len(body)
    y0 = CY + HH + 19 + (3 - n) * 6.5
    p.append(lines(cx, y0, body, 9.0, INK2, 13, anchor="middle"))

# ---- flow arrows between cards ----
ay = CY + CH / 2
for i in range(len(stages) - 1):
    p.append(f'<line x1="{centers[i][0]+cw+1.5}" y1="{ay}" x2="{centers[i+1][0]-2.5}" y2="{ay}" '
             f'stroke="#9098A2" stroke-width="2.1" marker-end="url(#ahg)"/>')

# ---- integrated example (real corpus item) ----
p.append(txt(20, 178, "What “Refine” does — self-contained rewrite "
             "(real corpus item, PMID 40578802):", 10, GRAY, "normal", italic=True))
EBY, EBH = 186, 66
p.append(rrect(20, EBY, 268, EBH, 8, CARD, "#C7CCD4", 1.3, shadow=True))
p.append(txt(32, EBY + 18, "original (extracted)", 8.8, GRAY, "bold"))
p.append(lines(32, EBY + 36, ["“The pathophysiology of PSVD", "remains unclear”"], 10, INK, 15, italic=True))
p.append(rrect(432, EBY, 268, EBH, 8, CARD, TEAL, 1.5, shadow=True))
p.append(txt(444, EBY + 18, "self-contained question", 8.8, TEAL, "bold"))
p.append(lines(444, EBY + 35, ["“What is the pathophysiologic", "mechanism of portosinusoidal",
                               "vascular disorder (PSVD)?”"], 9.6, INK, 13.5, italic=True))
ax_y = EBY + EBH / 2
p.append(f'<line x1="290" y1="{ax_y}" x2="430" y2="{ax_y}" stroke="{VERM}" '
         f'stroke-width="2.4" marker-end="url(#ahv)"/>')
p.append(txt(360, ax_y - 19, "self-containment", 9, VERM, "bold", "middle"))
p.append(txt(360, ax_y - 8, f"{SC_BEFORE}% → {SC_AFTER}%", 9, VERM, "bold", "middle"))

# ---- two additive tracks ----
p.append(txt(20, 268, "then layered on additively:", 9.5, GRAY, "normal", italic=True))
TBY, TBH = 274, 70
p.append(rrect(20, TBY, 333, TBH, 8, "#EAF3FA", BLUE, 1.4, shadow=True))
p.append(txt(34, TBY + 19, "(a)  Retrieval-grounded openness", 11, BLUE, "bold"))
p.append(lines(34, TBY + 33, [
    "status verifier → Stage-2 judge",
    "source-framing: 0 answered/unknown",
    f"grounded: {FLIP_PCT}% labels flip, {UNK_PCT}% unknown",
    f"0/{N_CORE} core questions still resolved"], 8.8, INK, 11))
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
