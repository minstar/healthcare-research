#!/usr/bin/env python3
"""New figure: the OpenBioRQ agentic evaluation harness, as a hand-authored SVG -> vector PDF.

Design ref: Gemini "Agent Execution" mockup, but with ACCURATE content only (the mockups'
labels are AI artifacts). Flow: open question -> agent doing multi-round tool use over the
ten real medical REST APIs (loop) -> answer+citations -> frozen per-question checklist judge
-> (deferred) human-expert agreement. blue->magenta gradient headers + flat icons. cairo embeds
subsetted TrueType (Type3=0).

Run with the torchtitan env python; writes paper_writing/figures/fig_evalflow.pdf.
"""
import cairosvg

ROOT = "/data/project/private/minstar/workspace/healthcare-research"
FIG = f"{ROOT}/paper_writing/figures"

BLUE, PURPLE, MAG = "#4A86C2", "#8A5CB0", "#C7508F"
CARD, INK, INK2, GRAY = "#FFFFFF", "#242424", "#5A5A5A", "#868C95"
CHIP = "#EFE7F4"
FONT = "DejaVu Sans"
TOOLS = ["PubMed", "ClinicalTrials", "OpenFDA", "Open Targets", "ChEMBL",
         "UniProt", "PubChem", "KEGG", "NCBI Datasets", "BioMCP"]

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


def rrect(x, y, w, h, rx, fill, stroke=None, sw=1.3, shadow=False, dash=None):
    s = ""
    if shadow:
        s += (f'<rect x="{x+1.5}" y="{y+2.4}" width="{w}" height="{h}" rx="{rx}" '
              f'fill="#26313B" opacity="0.13" filter="url(#blur)"/>')
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ''
    da = f' stroke-dasharray="{dash}"' if dash else ''
    s += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{st}{da}/>'
    return s


def top_round(x, y, w, h, r, fill):
    return (f'<path d="M{x},{y+r} a{r},{r} 0 0 1 {r},-{r} h{w-2*r} '
            f'a{r},{r} 0 0 1 {r},{r} v{h-r} h{-w} z" fill="{fill}"/>')


def card(x, y, w, h, hdr, title, ic):
    s = rrect(x, y, w, h, 9, CARD, "#D7DCE3", 1.0, shadow=True)
    s += top_round(x, y, w, 26, 9, hdr)
    s += ic(x + 16, y + 13)
    s += txt(x + 29, y + 17, title, 10.5, "#FFFFFF", "bold")
    return s


# icons (white on header)
def ic_q(cx, cy, c="#fff"):
    return (f'<circle cx="{cx}" cy="{cy}" r="8" fill="none" stroke="{c}" stroke-width="1.5"/>'
            f'<text x="{cx}" y="{cy+3.5}" font-family="{FONT}" font-size="10" font-weight="bold" '
            f'fill="{c}" text-anchor="middle">?</text>')


def ic_agent(cx, cy, c="#fff"):
    # robot head: rounded square + eyes + antenna
    return (f'<line x1="{cx}" y1="{cy-9}" x2="{cx}" y2="{cy-6}" stroke="{c}" stroke-width="1.4"/>'
            f'<circle cx="{cx}" cy="{cy-10}" r="1.4" fill="{c}"/>'
            f'<rect x="{cx-7}" y="{cy-6}" width="14" height="12" rx="3" fill="none" stroke="{c}" stroke-width="1.5"/>'
            f'<circle cx="{cx-3}" cy="{cy}" r="1.5" fill="{c}"/><circle cx="{cx+3}" cy="{cy}" r="1.5" fill="{c}"/>')


def ic_check(cx, cy, c="#fff"):
    # clipboard with check
    return (f'<rect x="{cx-6}" y="{cy-8}" width="12" height="16" rx="2" fill="none" stroke="{c}" stroke-width="1.4"/>'
            f'<rect x="{cx-3}" y="{cy-10}" width="6" height="3" rx="1" fill="{c}"/>'
            f'<path d="M{cx-3},{cy} l2,2.5 l4,-5" fill="none" stroke="{c}" stroke-width="1.5" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')


def ic_person(cx, cy, c="#fff"):
    return (f'<circle cx="{cx}" cy="{cy-4}" r="3.6" fill="none" stroke="{c}" stroke-width="1.5"/>'
            f'<path d="M{cx-6},{cy+8} a6,6 0 0 1 12,0" fill="none" stroke="{c}" stroke-width="1.5"/>')


p.append('<defs>')
p.append('<filter id="blur" x="-40%" y="-40%" width="180%" height="180%">'
         '<feGaussianBlur stdDeviation="2.1"/></filter>')
for mid, col in [("ahg", "#9098A2"), ("ahp", PURPLE)]:
    p.append(f'<marker id="{mid}" markerWidth="9" markerHeight="9" refX="6.5" refY="3" '
             f'orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L7,3 L0,6 Z" fill="{col}"/></marker>')
p.append('</defs>')

# title
p.append(txt(W / 2, 18, "Agentic evaluation harness", 13, INK, "bold", "middle"))

CY, CH = 30, 168
# --- block 1: open question ---
b1x, b1w = 10, 132
p.append(card(b1x, CY, b1w, CH, BLUE, "Open question", ic_q))
p.append(lines(b1x + 12, CY + 52, ["no answer key", "", "agent must gather", "and cite its own",
                                   "evidence"], 8.4, INK2, 14))
# --- block 2: agent + tool-use loop (big) ---
b2x, b2w = b1x + b1w + 26, 332
p.append(card(b2x, CY, b2w, CH, PURPLE, "Agent — multi-round tool use", ic_agent))
# agent glyph + loop on the left of the panel
agx, agy = b2x + 42, CY + 96
p.append(f'<circle cx="{agx}" cy="{agy}" r="22" fill="#F3ECF7" stroke="{PURPLE}" stroke-width="1.4"/>')
p.append(ic_agent(agx, agy, PURPLE))
# loop arrow (3/4 circle) around the agent
p.append(f'<path d="M{agx+26},{agy-6} a30,30 0 1 1 -10,-20" fill="none" stroke="{PURPLE}" '
         f'stroke-width="1.8" marker-end="url(#ahp)"/>')
p.append(txt(agx, agy + 40, "reason → act → observe", 7, PURPLE, "bold", "middle"))
# 10 tool chips, 2 cols x 5 rows on the right
tx0, ty0, cw_, ch_, gx_, gy_ = b2x + 92, CY + 40, 108, 17, 12, 5
for i, t in enumerate(TOOLS):
    col, row = divmod(i, 5)
    cx = tx0 + col * (cw_ + gx_)
    cy = ty0 + row * (ch_ + gy_)
    p.append(rrect(cx, cy, cw_, ch_, 8, CHIP, "#CDBBDB", 0.9))
    p.append(txt(cx + cw_ / 2, cy + 11.5, t, 7.2, "#5B3E73", "bold", "middle"))
p.append(txt(tx0 + cw_ + gx_ / 2, CY + CH - 8, "10 medical REST APIs", 7, GRAY, "normal", "middle"))
# --- block 3: checklist judge ---
b3x, b3w = b2x + b2w + 26, 174
p.append(card(b3x, CY, b3w, CH, MAG, "Checklist judge", ic_check))
p.append(lines(b3x + 12, CY + 50, [
    "answer + citations graded",
    "against a frozen",
    "per-question rubric",
    "", "solve = score ≥ 0.5",
    "inter-judge ρ 0.35 → 0.82"], 8.2, INK2, 13.5))
# --- arrows between blocks ---
ay = CY + CH / 2
p.append(f'<line x1="{b1x+b1w+2}" y1="{ay}" x2="{b2x-3}" y2="{ay}" stroke="#9098A2" stroke-width="2.1" marker-end="url(#ahg)"/>')
p.append(f'<line x1="{b2x+b2w+2}" y1="{ay}" x2="{b3x-3}" y2="{ay}" stroke="#9098A2" stroke-width="2.1" marker-end="url(#ahg)"/>')
p.append(txt((b2x + b2w + b3x) / 2, ay - 6, "answer", 6.6, GRAY, "normal", "middle"))

# --- deferred human-expert strip ---
sy = CY + CH + 14
p.append(rrect(10, sy, W - 20, 40, 8, "#F4F6F8", "#C2C7CF", 1.0, dash="4,3"))
p.append(ic_person(34, sy + 20, "#7A8390"))
p.append(txt(50, sy + 17, "Human-expert agreement", 9.5, INK, "bold"))
p.append(txt(50, sy + 31, "deferred (LLM-vs-LLM inter-judge reported now; human-expert validation in Limitations)",
             8, INK2))

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hgt}" '
       f'viewBox="0 0 {W} {Hgt}">' + "".join(p) + '</svg>')
out = f"{FIG}/fig_evalflow.pdf"
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=out)
print(f"wrote {out}")
