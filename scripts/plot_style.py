r"""
One place for the look of every figure: Arial everywhere, high resolution.

    import plot_style
    family = plot_style.use_arial()          # call once, before plotting

ARIAL ON UBUNTU
    Arial is a Microsoft font and is not installed on Ubuntu by default. Either:
      1. sudo apt install ttf-mscorefonts-installer          (accept the licence screen with Tab + Enter)
      2. or copy it from your Windows PC (Windows CMD, one line):
            scp C:\Windows\Fonts\arial*.ttf  USER@HOST:~/.local/share/fonts/
         (make the folder first on Ubuntu:  mkdir -p ~/.local/share/fonts)
    This module looks for arial*.ttf itself (~/.local/share/fonts, ~/.fonts,
    /usr/share/fonts and the folder in the environment variable MGN_FONT_DIR),
    so no matplotlib cache clearing is needed. If Arial is still missing it
    falls back to Liberation Sans (same letter widths as Arial) and says so.

PDF / SVG
    Fonts are embedded as TrueType (pdf.fonttype 42), the form journals ask for,
    and text stays editable in Illustrator / Inkscape.
"""

import glob
import os

import matplotlib
from matplotlib import font_manager

DPI = 300                       # default resolution of every saved PNG
FALLBACKS = ("Liberation Sans", "Helvetica", "DejaVu Sans")
_FAMILY = None


def _register_arial(extra_dirs=()):
    dirs = list(extra_dirs) + [os.environ.get("MGN_FONT_DIR", ""), "~/.local/share/fonts", "~/.fonts",
                                "/usr/share/fonts/truetype/msttcorefonts", "/usr/share/fonts",
                                "/usr/local/share/fonts", "C:/Windows/Fonts"]
    found = []
    for d in dirs:
        d = os.path.expanduser(d) if d else ""
        if not d or not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "**", "*.[tT][tT][fF]"), recursive=True):
            if os.path.basename(p).lower().startswith("arial"):
                try:
                    font_manager.fontManager.addfont(p)
                    found.append(p)
                except Exception:
                    pass
        if found:
            break
    return found


def use_arial(extra_dirs=(), quiet=False):
    """Set Arial (or the closest fallback) for all text, maths included. Returns the family used."""
    global _FAMILY
    if _FAMILY is not None:
        return _FAMILY
    names = set(f.name for f in font_manager.fontManager.ttflist)
    if "Arial" not in names:
        _register_arial(extra_dirs)
        names = set(f.name for f in font_manager.fontManager.ttflist)
    if "Arial" in names:
        family = "Arial"
    else:
        family = next((f for f in FALLBACKS if f in names), "DejaVu Sans")
        if not quiet:
            print("NOTE: Arial is not installed - the figures use %s instead. To get Arial:\n"
                  "      sudo apt install ttf-mscorefonts-installer\n"
                  "      (or copy C:\\Windows\\Fonts\\arial*.ttf to ~/.local/share/fonts/ - see scripts/plot_style.py)"
                  % family)
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family] + [f for f in ("Arial",) + FALLBACKS if f != family],
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": family + ":italic",
        "mathtext.bf": family + ":bold",
        "mathtext.cal": family,                  # no "cursive font not found" messages
        "mathtext.sf": family,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": DPI,
        "figure.dpi": 100,
    })
    _FAMILY = family
    return family


def save(fig, path, formats=("png",), dpi=None, **kw):
    """Save one figure in every requested format: path 'a/b/name.png' -> name.png, name.pdf ..."""
    stem = os.path.splitext(path)[0]
    out = []
    for f in formats:
        p = "%s.%s" % (stem, f)
        fig.savefig(p, dpi=dpi or DPI, facecolor="white", **kw)
        out.append(p)
    return out
