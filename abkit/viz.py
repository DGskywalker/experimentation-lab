"""Shared matplotlib style so every figure in the project reads as one system."""

# matplotlib is imported inside the functions so the palette can be used (e.g. by the
# Streamlit app) without paying matplotlib's import and font-cache cost.

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#a3a29c"
CONTROL = "#2a78d6"    # categorical slot 1
TREATMENT = "#eb6834"  # categorical slot 2
THIRD = "#1baf7a"      # categorical slot 3
CRITICAL = "#c8312f"   # status: reserved for "bad" thresholds, always labelled


def use_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
    })


def save(fig, path) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(path)
    plt.close(fig)
