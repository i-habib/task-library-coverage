#!/usr/bin/env python3
"""Render the learned-temporal endpoint audit as native PGFPlots source."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


COLORS = {"256": "orange!85!black", "4096": "blue!75!black"}


def error_coordinate(row: dict, field: str) -> str:
    value = row[field]["mean"]
    lower = value - row[field]["lower"]
    upper = row[field]["upper"] - value
    return f"  ({row['prefix']},{value:.6f}) += (0,{upper:.6f}) -= (0,{lower:.6f})"


def distance_ratio_coordinate(row: dict) -> str:
    """Transform the saved S bootstrap interval monotonically to Delta_D/Delta_C."""
    values = {key: (1 + row["S"][key]) / (1 - row["S"][key])
              for key in ("lower", "mean", "upper")}
    lower = values["mean"] - values["lower"]
    upper = values["upper"] - values["mean"]
    return f"  ({row['prefix']},{values['mean']:.6f}) += (0,{upper:.6f}) -= (0,{lower:.6f})"


def render(report: dict) -> str:
    all_rows = [row for arm in report["arms"].values() for row in arm["prefixes"]]
    ratio_high = max(row["risk_ratio"]["upper"] for row in all_rows)
    ratio_low = min(row["risk_ratio"]["lower"] for row in all_rows)
    risk_ymin = math.floor((min(0.99, ratio_low) - 0.03) * 20) / 20
    risk_ymax = math.ceil((max(1.2, ratio_high) + 0.03) * 20) / 20

    lines = [
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=2 by 1,horizontal sep=1.0cm},",
        r"  width=0.48\linewidth,height=4.15cm,",
        r"  tick label style={font=\scriptsize},",
        r"  label style={font=\scriptsize},",
        r"  legend style={font=\scriptsize,draw=none,fill=none},",
        r"  grid=major,grid style={black!10},",
        r"  axis line style={black!65},tick style={black!65}",
        r"]",
        r"\nextgroupplot[",
        r"  xlabel={context transitions $k$},",
        r"  ylabel={risk / continuous Bayes},",
        f"  xmin=2,xmax=16,ymin={risk_ymin:.2f},ymax={risk_ymax:.2f},",
        r"  xtick={3,7,11,15},",
        r"  legend pos=north east",
        r"]",
        r"\addplot[black!55,dashed,domain=2:16] {1.2};",
    ]
    for key in ("256", "4096"):
        rows = report["arms"][key]["prefixes"]
        tex_m = f"{int(key):,}".replace(",", "{,}")
        lines.extend([
            rf"\addplot+[{COLORS[key]},mark=*,thick,error bars/.cd,y dir=both,y explicit]",
            "coordinates {",
            *[error_coordinate(row, "risk_ratio") for row in rows],
            "};",
            rf"\addlegendentry{{$M={tex_m}$}}",
        ])
    lines.extend([
        r"\nextgroupplot[",
        r"  xlabel={context transitions $k$},",
        r"  ylabel={$\Delta_D/\Delta_C$ (log scale)},",
        r"  xmin=2,xmax=16,ymin=0.7,ymax=130,",
        r"  xtick={3,7,11,15},ymode=log,ytick={1,10,100},",
        r"  yticklabels={$1\!\times$,$10\!\times$,$100\!\times$}",
        r"]",
        r"\addplot[black!55,dashed,domain=2:16] {1};",
    ])
    for key in ("256", "4096"):
        rows = report["arms"][key]["prefixes"]
        lines.extend([
            rf"\addplot+[{COLORS[key]},mark=*,thick,error bars/.cd,y dir=both,y explicit]",
            "coordinates {",
            *[distance_ratio_coordinate(row) for row in rows],
            "};",
        ])
    lines.extend([
        r"\node[font=\tiny,anchor=south east,align=right] at (axis cs:15,1.28) {$G=10.84$ bits};",
        r"\node[font=\tiny,anchor=north east,align=right] at (axis cs:15,61) {$G=6.84$ bits};",
    ])
    lines.extend([r"\end{groupplot}", r"\end{tikzpicture}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.audit.read_text())
    args.output.write_text(render(report))


if __name__ == "__main__":
    main()
