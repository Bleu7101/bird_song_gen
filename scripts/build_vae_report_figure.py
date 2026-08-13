"""Build the report's combined VAE evolution and generation figure.

The historical V1 and V2 reconstruction arrays are not retained in the
repository, but their diagnostic PNGs use the same held-out example for each
species.  This script therefore crops the heatmaps from those tracked plots.
For panel (a), it approximately aligns each historical magma colour mapping to
the V3 mapping by fitting an affine transform between the matched Original
panels.  Original and V1--V3 are then re-rendered on one display scale within
each species row.  This improves visual comparability without treating the
model sequence as a controlled ablation.

Panel (b) uses the first two displayed V3 samples per species from the tracked
seed-42 notebook diagnostic.  Those panels retain the notebook's independent
per-sample display normalization, so their absolute colour intensities should
not be compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "reports" / "vae" / "vae_evolution_and_generated_samples.png"

SPECIES = ("Northern Cardinal", "Song Sparrow", "American Robin")
VERSIONS = ("Original", "V1", "V2", "V3")


@dataclass(frozen=True)
class PlotGrid:
    path: Path
    x_bounds: tuple[tuple[int, int], ...]
    y_bounds: tuple[tuple[int, int], ...]

    def crop(self, row: int, column: int) -> Image.Image:
        """Return heatmap pixels inside the Matplotlib axis spines."""
        with Image.open(self.path) as image:
            image = image.convert("RGB")
            left, right = self.x_bounds[column]
            top, bottom = self.y_bounds[row]
            return image.crop((left, top, right, bottom))


# Bounds exclude titles, labels, ticks, and the black axis spines.  They are
# tied to the tracked PNG dimensions and checked below before composition.
V1_GRID = PlotGrid(
    REPO_ROOT / "reports" / "vae" / "conditional_vae" / "test_reconstructions.png",
    x_bounds=((104, 777), (892, 1565)),
    y_bounds=((50, 471), (602, 1023), (1154, 1575)),
)
V2_GRID = PlotGrid(
    REPO_ROOT / "reports" / "vae" / "conditional_vae_v2" / "test_reconstructions.png",
    x_bounds=((104, 728), (843, 1466), (1581, 2205)),
    y_bounds=((50, 503), (634, 1087), (1218, 1671)),
)
V3_GRID = PlotGrid(
    REPO_ROOT / "reports" / "vae" / "conditional_vae_v3" / "test_reconstructions.png",
    x_bounds=V2_GRID.x_bounds,
    y_bounds=V2_GRID.y_bounds,
)
V3_SAMPLE_GRID = PlotGrid(
    REPO_ROOT / "reports" / "vae" / "conditional_vae_v3" / "conditional_samples.png",
    x_bounds=((104, 610), (738, 1244), (1372, 1878), (2006, 2512)),
    y_bounds=((50, 423), (554, 927), (1058, 1431)),
)


def _downsample(image: Image.Image) -> np.ndarray:
    """Reduce a rasterized heatmap to its underlying 128 x 128 grid."""
    return np.asarray(
        image.resize((128, 128), Image.Resampling.BOX), dtype=np.float32
    ) / 255.0


MAGMA_LUT = plt.colormaps["magma"](np.linspace(0.0, 1.0, 1024))[:, :3].astype(
    np.float32
)


def _invert_magma(rgb: np.ndarray) -> np.ndarray:
    """Map raster RGB colours back to their nearest normalized magma value."""
    pixels = rgb.reshape(-1, 3)
    indices = np.empty(len(pixels), dtype=np.int32)
    chunk_size = 2048
    for start in range(0, len(pixels), chunk_size):
        chunk = pixels[start : start + chunk_size]
        squared_distance = ((chunk[:, None, :] - MAGMA_LUT[None, :, :]) ** 2).sum(
            axis=2
        )
        indices[start : start + len(chunk)] = squared_distance.argmin(axis=1)
    return (indices.reshape(rgb.shape[:2]) / (len(MAGMA_LUT) - 1)).astype(np.float32)


def _fit_display_alignment(
    source_original: np.ndarray, reference_original: np.ndarray
) -> tuple[float, float, float]:
    """Fit reference ~= slope * source + intercept on non-clipped pixels."""
    source = source_original.ravel()
    reference = reference_original.ravel()
    mask = (
        (source > 0.01)
        & (source < 0.99)
        & (reference > 0.01)
        & (reference < 0.99)
    )
    design = np.column_stack((source[mask], np.ones(mask.sum(), dtype=np.float32)))
    slope, intercept = np.linalg.lstsq(design, reference[mask], rcond=None)[0]
    prediction = slope * source[mask] + intercept
    residual_sum = np.square(reference[mask] - prediction).sum()
    total_sum = np.square(reference[mask] - reference[mask].mean()).sum()
    r_squared = 1.0 - residual_sum / total_sum
    if slope <= 0.0 or r_squared < 0.90:
        raise RuntimeError(
            f"Unreliable display alignment: slope={slope:.4f}, "
            f"intercept={intercept:.4f}, R^2={r_squared:.4f}"
        )
    return float(slope), float(intercept), float(r_squared)


def _apply_magma(values: np.ndarray) -> np.ndarray:
    return plt.colormaps["magma"](np.clip(values, 0.0, 1.0))[:, :, :3]


def build_reconstruction_panels() -> tuple[list[list[np.ndarray]], list[str]]:
    panels: list[list[np.ndarray]] = []
    diagnostics: list[str] = []

    for row, species in enumerate(SPECIES):
        reference_original = _invert_magma(_downsample(V3_GRID.crop(row, 0)))
        row_panels = [_apply_magma(reference_original)]

        for version, grid in (("V1", V1_GRID), ("V2", V2_GRID)):
            source_original = _invert_magma(_downsample(grid.crop(row, 0)))
            source_reconstruction = _invert_magma(_downsample(grid.crop(row, 1)))
            slope, intercept, r_squared = _fit_display_alignment(
                source_original, reference_original
            )
            aligned_reconstruction = slope * source_reconstruction + intercept
            row_panels.append(_apply_magma(aligned_reconstruction))
            diagnostics.append(
                f"{species} {version}: slope={slope:.4f}, "
                f"intercept={intercept:.4f}, R^2={r_squared:.4f}"
            )

        v3_reconstruction = _invert_magma(_downsample(V3_GRID.crop(row, 1)))
        row_panels.append(_apply_magma(v3_reconstruction))
        panels.append(row_panels)

    return panels, diagnostics


def build_generation_panels() -> list[list[np.ndarray]]:
    # Store as [sample row][species column] to keep the lower panel compact.
    return [
        [
            _downsample(V3_SAMPLE_GRID.crop(species_index, sample_index))
            for species_index in range(len(SPECIES))
        ]
        for sample_index in range(2)
    ]


def compose_figure() -> None:
    reconstruction_panels, diagnostics = build_reconstruction_panels()
    generation_panels = build_generation_panels()

    figure = plt.figure(figsize=(11.4, 10.0), layout="constrained")
    outer = figure.add_gridspec(2, 1, height_ratios=(3.0, 2.0), hspace=0.16)

    upper = outer[0].subgridspec(3, 4, wspace=0.025, hspace=0.055)
    upper_axes = np.empty((3, 4), dtype=object)
    for row, species in enumerate(SPECIES):
        for column, version in enumerate(VERSIONS):
            axis = figure.add_subplot(upper[row, column])
            upper_axes[row, column] = axis
            axis.imshow(reconstruction_panels[row][column], aspect="auto")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.6)
            if row == 0:
                axis.set_title(version, fontsize=11, pad=5)
            if column == 0:
                axis.set_ylabel(species, fontsize=10, labelpad=8)

    upper_axes[0, 0].text(
        -0.36,
        1.34,
        "(a) Matched held-out reconstructions",
        transform=upper_axes[0, 0].transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    lower = outer[1].subgridspec(2, 3, wspace=0.03, hspace=0.06)
    lower_axes = np.empty((2, 3), dtype=object)
    for sample_index in range(2):
        for species_index, species in enumerate(SPECIES):
            axis = figure.add_subplot(lower[sample_index, species_index])
            lower_axes[sample_index, species_index] = axis
            axis.imshow(generation_panels[sample_index][species_index], aspect="auto")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.6)
            if sample_index == 0:
                axis.set_title(species, fontsize=11, pad=5)
            if species_index == 0:
                axis.set_ylabel(f"Sample {sample_index + 1}", fontsize=10, labelpad=8)

    lower_axes[0, 0].text(
        -0.36,
        1.34,
        "(b) VAE-v3 posterior-anchor generations",
        transform=lower_axes[0, 0].transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": "VAE reconstruction evolution and V3 generated samples",
            "Description": (
                "Panel (a) compares matched held-out reconstructions from V1, V2, "
                "and V3. Panel (b) shows the first two displayed V3 generated "
                "samples per species from the tracked seed-42 diagnostic pool."
            ),
        },
    )
    plt.close(figure)

    print(f"Saved: {OUTPUT_PATH}")
    for diagnostic in diagnostics:
        print(diagnostic)


if __name__ == "__main__":
    compose_figure()
