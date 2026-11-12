"""The one chart for the README: daily returns, VaR lines and breaches."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Colours from the portfolio site
NAVY = "#1d2f4a"
RED = "#8c2a37"
SAGE = "#4c7a62"
GREY = "#9aa3ad"
INK = "#1f2933"
MUTED = "#5b6670"


def plot_var_breaches(forecasts, level=99, start=None, end=None, path=None):
    """Plot daily returns against the historical and Monte Carlo VaR lines.

    VaR is drawn as a negative return (the loss threshold). Breach days, when
    the return fell below the line, are marked: red circles for historical
    simulation and green triangles for Monte Carlo.
    """
    f = forecasts.loc[start:end]
    hist = f[f"VaR{level} Historical"]
    mc = f[f"VaR{level} Monte Carlo (EWMA-t)"]
    ret = f["return"]

    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
    ax.vlines(f.index, 0, ret, color=GREY, linewidth=0.6, alpha=0.8, label="Daily log return")
    ax.plot(f.index, -hist, color=NAVY, linewidth=1.6, label=f"{level}% VaR, historical simulation")
    ax.plot(f.index, -mc, color=SAGE, linewidth=1.6, linestyle="--",
            label=f"{level}% VaR, Monte Carlo (EWMA vol, t shocks)")

    hist_hits = ret < -hist
    mc_hits = ret < -mc
    ax.scatter(f.index[hist_hits], ret[hist_hits], s=42, color=RED, edgecolor="white",
               linewidth=0.8, zorder=5, label=f"Breach, historical ({hist_hits.sum()})")
    ax.scatter(f.index[mc_hits], ret[mc_hits], s=46, marker="^", facecolor="none",
               edgecolor=SAGE, linewidth=1.4, zorder=6, label=f"Breach, Monte Carlo ({mc_hits.sum()})")

    ax.axhline(0, color=MUTED, linewidth=0.6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Daily log return", color=INK)
    ax.set_title(f"Henry Hub natural gas: daily returns vs. 1-day {level}% VaR "
                 f"({f.index[0]:%b %Y} to {f.index[-1]:%b %Y})",
                 loc="left", color=INK, fontsize=12.5, fontweight="bold")
    ax.grid(axis="y", color="#e3e7eb", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c7ced5")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.07), fontsize=8.5, frameon=False, ncol=3)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    return fig, ax
