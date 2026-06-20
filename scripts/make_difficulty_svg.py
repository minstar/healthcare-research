#!/usr/bin/env python3
import os
"""Fig 3 (empirical-difficulty analysis) as a hand-authored SVG -> vector PDF via cairosvg.

Design ref: the externally-drawn mockup (figure3_figure4_combined.png, top panel) — a colored
section header + two soft cards: (a) the question-granular difficulty-bucket stack on the priority
track and (b) the per-roster-model mean checklist score against the 0.5 pass line. All numbers are
FACTS-only (priority track n=525: 256/236/33; avg checklist GLM 0.324 / Qwen 0.447 / DSV4 0.309).
cairo embeds subsetted TrueType (not Type3) -> AAAI font-compliant.

Run with the torchtitan env python; writes paper_writing/figures/fig_difficulty.pdf.
"""
import cairosvg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = f"{ROOT}/paper_writing/figures"

# palette (consistent with the other OpenBioRQ figures)
BLUE, TEAL, GOLD, VERM = "#5A9BD4", "#46B08F", "#E8B85E", "#E0875A"
CORE, DISC, EASY = "#E0726A", "#E8B85E", "#46B08F"   # core(red) / discriminating(gold) / easy(teal)
HDR = "#6E62B6"                                       # purple section header (matches mockup)
CARD, BAND = "#FFFFFF", "#F5F8FB"
INK, INK2, GRAY, LABEL = "#242424", "#5A5A5A", "#868C95", "#5A6473"
FONT = "DejaVu Sans"

# ---- FACTS ----
N = 525
BUCKETS = [("Core", "All-3 fail", 256, 49, CORE),
           ("Discriminating", "Split", 236, 45, DISC),
           ("Easy", "All-3 pass", 33, 6, EASY)]
HARD_PCT = 94
MODELS = [("GLM-5.1", 0.324), ("Qwen3.6", 0.447), ("DeepSeek-V4", 0.309)]
PASS = 0.5

W, Hgt = 720, 286
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
              f'fill="#26313B" opacity="0.13" filter="url(#blur)"/>')
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ''
    s += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{st}/>'
    return s


def chip(x, y, col, label, sub):
    s = f'<rect x="{x}" y="{y-8}" width="11" height="11" rx="2.5" fill="{col}"/>'
    s += txt(x + 16, y - 1.5, label, 9.3, INK, "bold")
    s += txt(x + 16, y + 9, sub, 8.2, GRAY)
    return s


p.append('<defs>')
p.append('<filter id="blur" x="-40%" y="-40%" width="180%" height="180%">'
         '<feGaussianBlur stdDeviation="2.3"/></filter>')
p.append('</defs>')
p.append(f'<rect x="0" y="0" width="{W}" height="{Hgt}" fill="#FFFFFF"/>')

# ---- section header bar ----
p.append(rrect(6, 6, W - 12, 30, 9, HDR))
p.append(txt(20, 26, "Empirical Difficulty Analysis", 14, "#FFFFFF", "bold"))
p.append(txt(W - 20, 25, "priority_setting track · n=525 · question-granular", 9.5,
             "#E7E3F4", "normal", "end"))

# ================= panel (a): difficulty buckets =================
AX, AY, AW, AH = 6, 44, 350, 236
p.append(rrect(AX, AY, AW, AH, 10, CARD, "#D7DCE3", 1.0, shadow=True))
p.append(txt(AX + 16, AY + 24, "(a)  Difficulty buckets", 11.5, INK, "bold"))
p.append(txt(AX + 16, AY + 39, "how many of the 3 roster models fail each question", 8.6, GRAY))

# stacked bar
bx, by, bw, bh = AX + 18, AY + 58, AW - 36, 38
xacc = bx
for name, sub, cnt, pct, col in BUCKETS:
    seg = bw * cnt / N
    p.append(f'<rect x="{xacc:.1f}" y="{by}" width="{seg:.1f}" height="{bh}" fill="{col}"/>')
    if seg > 34:
        p.append(txt(xacc + seg / 2, by + bh / 2 - 1, str(cnt), 12.5, "#FFFFFF", "bold", "middle"))
        p.append(txt(xacc + seg / 2, by + bh / 2 + 12, f"{pct}%", 8.8, "#FFFFFF", "bold", "middle"))
    else:
        p.append(txt(xacc + seg / 2, by - 5, f"{cnt}", 8.5, INK, "bold", "middle"))
        p.append(txt(xacc + seg / 2, by + bh + 11, f"{pct}%", 8.2, INK2, "bold", "middle"))
    xacc += seg
# rounded clip frame on the bar
p.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="5" fill="none" '
         f'stroke="#FFFFFF" stroke-width="2.4"/>')

# callout pill
cpy = by + bh + 26
p.append(rrect(bx, cpy, bw, 26, 13, "#FBEDE9", CORE, 1.2))
p.append(txt(bx + bw / 2, cpy + 17, f"{HARD_PCT}% are hard for ≥1 roster model",
             10.5, "#B5483F", "bold", "middle"))

# legend
ly = cpy + 56
for i, (name, sub, cnt, pct, col) in enumerate(BUCKETS):
    p.append(chip(bx + 2, ly + i * 24, col, f"{name}  ({sub})", f"{cnt} questions  ·  {pct}%"))

# ================= panel (b): mean checklist score =================
BXp, BYp, BWp, BHp = 366, 44, 348, 236
p.append(rrect(BXp, BYp, BWp, BHp, 10, CARD, "#D7DCE3", 1.0, shadow=True))
p.append(txt(BXp + 16, BYp + 24, "(b)  Mean checklist score per roster model", 11.5, INK, "bold"))
p.append(txt(BXp + 16, BYp + 39, "averaged over the priority track (n=525)", 8.6, GRAY))

# plot frame
plx, ply_top, plw, ply_bot = BXp + 52, BYp + 56, 250, BYp + 176
ymax = 0.6


def sy(score):
    return ply_bot - (score / ymax) * (ply_bot - ply_top)


# y gridlines + ticks
for tick in (0.0, 0.25, 0.5):
    yy = sy(tick)
    p.append(f'<line x1="{plx}" y1="{yy:.1f}" x2="{plx+plw}" y2="{yy:.1f}" '
             f'stroke="#E7EBF0" stroke-width="1"/>')
    p.append(txt(plx - 8, yy + 3.3, f"{tick:.2f}", 8.2, GRAY, "normal", "end"))
# axis
p.append(f'<line x1="{plx}" y1="{ply_top-6}" x2="{plx}" y2="{ply_bot}" stroke="#B9C0C9" stroke-width="1.2"/>')
p.append(f'<line x1="{plx}" y1="{ply_bot}" x2="{plx+plw}" y2="{ply_bot}" stroke="#B9C0C9" stroke-width="1.2"/>')

# pass threshold line
yp = sy(PASS)
p.append(f'<line x1="{plx}" y1="{yp:.1f}" x2="{plx+plw}" y2="{yp:.1f}" stroke="{VERM}" '
         f'stroke-width="1.8" stroke-dasharray="6 3"/>')
p.append(txt(plx + plw, yp - 5, "pass threshold 0.5", 8.6, VERM, "bold", "end"))

# model dots
n = len(MODELS)
step = plw / (n + 1)
mcol = [BLUE, GOLD, TEAL]
for i, (name, score) in enumerate(MODELS):
    cx = plx + step * (i + 1)
    cyv = sy(score)
    p.append(f'<line x1="{cx:.1f}" y1="{ply_bot}" x2="{cx:.1f}" y2="{cyv:.1f}" '
             f'stroke="{mcol[i]}" stroke-width="2" opacity="0.45"/>')
    p.append(f'<circle cx="{cx:.1f}" cy="{cyv:.1f}" r="7" fill="{mcol[i]}" stroke="#FFFFFF" stroke-width="1.6"/>')
    p.append(txt(cx, cyv - 12, f"{score:.2f}", 9.5, INK, "bold", "middle"))
    p.append(txt(cx, ply_bot + 14, name, 8.6, INK2, "bold", "middle"))

# note
ny = ply_bot + 34
p.append(rrect(BXp + 16, ny, BWp - 32, 26, 8, "#EAF3FA", BLUE, 1.1))
p.append(txt(BXp + BWp / 2, ny + 17, "all three score below 0.5 — the track is genuinely challenging",
             9.4, "#2F6CA0", "bold", "middle"))

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hgt}" '
       f'viewBox="0 0 {W} {Hgt}">' + "".join(p) + '</svg>')
out = f"{FIG}/fig_difficulty.pdf"
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=out)
print(f"wrote {out}")
