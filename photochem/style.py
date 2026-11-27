import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import FIGURES, TEST_FIGURES, load_config


def apply():
	style = load_config()[0]["style"]
	plt.rcParams.update({"font.size": style["font_size"], "savefig.dpi": style["dpi"], **style["rc"]})
	return style


def figure(width="column", aspect=0.75, **kwargs):
	"""Figure sized for the paper: width = "column" | "page" | inches."""
	style = apply()
	w = {"column": style["column_width"], "page": style["page_width"]}.get(width, width)
	return plt.subplots(figsize=(w, w * aspect), **kwargs)


def colorbar(fig, im, ax, label, location="right", **kwargs):
	"""Colorbar exactly as tall (location="right") or wide ("top") as `ax` (fig.colorbar(ax=ax) spans the space reserved for ax,
	which an equal-aspect image does not fill). kwargs go to fig.colorbar (e.g. extend="min")."""
	if location == "top":
		cbar = fig.colorbar(im, cax=ax.inset_axes([0.0, 1.04, 1.0, 0.05]), orientation="horizontal", location="top", **kwargs)
	else:
		cbar = fig.colorbar(im, cax=ax.inset_axes([1.04, 0.0, 0.05, 1.0]), **kwargs)
	cbar.set_label(label)
	return cbar


def save(fig, name, test=False):
	"""Save to the configured `figures` directory, or to test_figures/ for a small test that is not in the paper (test=True)."""
	out = TEST_FIGURES if test else FIGURES
	out.mkdir(parents=True, exist_ok=True)
	path = out / f"{name}.pdf"
	fig.savefig(path)
	print(f"wrote {path}")
	return path
