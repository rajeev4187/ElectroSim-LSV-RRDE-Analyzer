"""A simple Randles-circuit schematic (Rs + (Rct || Cdl)) for the EIS report.

Drawn with plain plotly shapes/annotations (no schemdraw dependency), styled
to match the other reviewer-response figures.
"""
import os
import sys

import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.make_reviewer_figures import FAMILY, OUT, ROOT, render_png  # noqa: E402

FONT = {"family": FAMILY, "size": 22, "color": "black"}
LW = 2.5


def _line(fig, x0, y0, x1, y1):
    fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                  line={"color": "black", "width": LW})


def _resistor(fig, x0, x1, y, label):
    """A zig-zag resistor symbol along a horizontal line at height y."""
    n_zig = 6
    span = x1 - x0
    lead = span * 0.15
    body0, body1 = x0 + lead, x1 - lead
    _line(fig, x0, y, body0, y)
    _line(fig, body1, y, x1, y)
    seg = (body1 - body0) / n_zig
    amp = 0.09
    pts_x = [body0]
    pts_y = [y]
    for i in range(n_zig):
        xm = body0 + seg * (i + 0.5)
        ym = y + (amp if i % 2 == 0 else -amp)
        pts_x.append(xm)
        pts_y.append(ym)
    pts_x.append(body1)
    pts_y.append(y)
    fig.add_trace(go.Scatter(x=pts_x, y=pts_y, mode="lines",
                             line={"color": "black", "width": LW},
                             showlegend=False, hoverinfo="skip"))
    fig.add_annotation(x=(x0 + x1) / 2, y=y + 0.32, text=label, showarrow=False,
                       font=FONT)


def _capacitor(fig, x, y0, y1, label):
    """A capacitor symbol (two parallel plates) along a vertical line at x."""
    mid = (y0 + y1) / 2
    gap = 0.12
    _line(fig, x, y0, x, mid - gap)
    _line(fig, x, mid + gap, x, y1)
    plate_w = 0.35
    _line(fig, x - plate_w / 2, mid - gap, x + plate_w / 2, mid - gap)
    _line(fig, x - plate_w / 2, mid + gap, x + plate_w / 2, mid + gap)
    fig.add_annotation(x=x + 0.55, y=mid, text=label, showarrow=False, font=FONT)


def main() -> None:
    fig = go.Figure()

    y_top, y_bot = 1.0, -1.0
    x_left, x_rs_end, x_branch, x_right = 0.0, 1.6, 3.2, 4.8

    # Working-electrode lead -> Rs -> node A
    _line(fig, x_left, 0, x_left + 0.3, 0)
    fig.add_annotation(x=x_left - 0.15, y=0, text="WE", showarrow=False,
                       font=FONT, xanchor="right")
    _resistor(fig, x_left + 0.3, x_rs_end, 0, "R_s")
    _line(fig, x_rs_end, 0, x_branch, 0)

    # Node A splits into two vertical branches: Rct (top) and Cdl (bottom),
    # rejoining at node B before the counter-electrode lead.
    _line(fig, x_branch, 0, x_branch, y_top)
    _line(fig, x_branch, 0, x_branch, y_bot)
    _resistor(fig, x_branch, x_right - 0.2, y_top, "")
    fig.add_annotation(x=(x_branch + x_right - 0.2) / 2, y=y_top + 0.32,
                       text="R_ct", showarrow=False, font=FONT)
    _capacitor(fig, (x_branch + x_right - 0.2) / 2, y_bot - 0.32, y_bot + 0.32,
              "")
    fig.add_annotation(x=(x_branch + x_right - 0.2) / 2, y=y_bot - 0.55,
                       text="C_dl", showarrow=False, font=FONT)
    _line(fig, x_right - 0.2, y_top, x_right, y_top)
    _line(fig, x_right - 0.2, y_bot, x_right, y_bot)
    _line(fig, x_right, y_top, x_right, y_bot)
    _line(fig, x_right, 0, x_right + 0.5, 0)
    fig.add_annotation(x=x_right + 0.65, y=0, text="CE", showarrow=False,
                       font=FONT, xanchor="left")
    # Redraw the node-B vertical bus that the two lines above only touched
    # at their ends (kept as a single clean line for the figure).

    # Nodes
    for x in (x_branch, x_right):
        fig.add_trace(go.Scatter(x=[x], y=[0], mode="markers",
                                 marker={"size": 7, "color": "black"},
                                 showlegend=False, hoverinfo="skip"))

    fig.update_xaxes(visible=False, range=[-0.9, 5.8])
    fig.update_yaxes(visible=False, range=[-1.9, 1.6], scaleanchor="x",
                     scaleratio=1)
    fig.update_layout(
        template="plotly_white", width=1150, height=620,
        paper_bgcolor="white", plot_bgcolor="white",
        margin={"l": 40, "r": 40, "t": 70, "b": 90},
        title={"text": "Randles equivalent circuit used for the Ru/Rct fits",
              "x": 0.5, "xanchor": "center",
              "font": {"family": FAMILY, "size": 24}},
    )
    fig.add_annotation(
        x=2.4, y=-1.75, showarrow=False, align="center",
        text=("R_s: uncompensated/solution resistance (high-frequency"
             " real-axis intercept)<br>R_ct: charge-transfer resistance"
             " (arc diameter)  ·  C_dl: double-layer capacitance"),
        font={"family": FAMILY, "size": 15, "color": "#444444"},
    )
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig_circuit_randles.png")
    render_png(fig, path, dpi=300)
    print(f"wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
