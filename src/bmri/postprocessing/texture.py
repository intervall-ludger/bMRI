from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops


@dataclass
class TextureConfig:
    """Configuration for GLCM texture analysis."""

    gray_levels: int = 16
    distances: list[int] = field(default_factory=lambda: [1])
    angles: list[float] = field(default_factory=lambda: [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    clip_range: tuple[float, float] | None = None
    features: list[str] = field(
        default_factory=lambda: ["contrast", "homogeneity", "energy", "variance"]
    )
    backend: str = "skimage"  # "skimage" or "pyradiomics"


# Features available per backend
SKIMAGE_FEATURES = {"contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM"}
MANUAL_FEATURES = {"variance", "entropy"}


class TextureAnalysis:
    """GLCM texture analysis on parameter maps, per ROI, per slice."""

    def __init__(self, config: TextureConfig | None = None):
        self.config = config or TextureConfig()

    def run(
        self,
        param_map: np.ndarray | Path,
        mask: np.ndarray | Path,
    ) -> pd.DataFrame:
        """Run texture analysis on a parameter map with ROI mask.

        Args:
            param_map: 3D array (x, y, z) or path to NIfTI
            mask: 3D integer array (x, y, z) with ROI labels, or path to NIfTI

        Returns:
            DataFrame with columns: roi, slice, feature_name, value
        """
        if isinstance(param_map, Path):
            param_map = nib.load(param_map).get_fdata()
        if isinstance(mask, Path):
            mask = nib.load(mask).get_fdata().round().astype(int)

        if self.config.backend == "pyradiomics":
            return self._run_pyradiomics(param_map, mask)
        return self._run_skimage(param_map, mask)

    def run_and_summarize(
        self,
        param_map: np.ndarray | Path,
        mask: np.ndarray | Path,
    ) -> pd.DataFrame:
        """Run texture analysis and aggregate over slices (mean per ROI)."""
        df = self.run(param_map, mask)
        if df.empty:
            return df
        return (
            df.groupby(["roi", "feature"])["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"count": "n_slices"})
        )

    def _run_skimage(self, param_map: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
        cfg = self.config
        rows = []

        rois = np.unique(mask)
        rois = rois[rois > 0]

        for s in range(param_map.shape[2]):
            param_slice = param_map[:, :, s].astype(np.float64)
            mask_slice = mask[:, :, s]

            for roi in rois:
                roi_mask = mask_slice == roi
                if np.sum(roi_mask) < 4:
                    continue

                features = self._compute_glcm_slice(param_slice, roi_mask, cfg)
                for fname, val in features.items():
                    rows.append({"roi": int(roi), "slice": s, "feature": fname, "value": val})

        return pd.DataFrame(rows)

    def _compute_glcm_slice(
        self, param_slice: np.ndarray, roi_mask: np.ndarray, cfg: TextureConfig
    ) -> dict[str, float]:
        """Compute GLCM features for a single ROI in a single slice."""
        # Extract ROI bounding box
        ys, xs = np.where(roi_mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        roi_patch = param_slice[y0:y1, x0:x1].copy()
        mask_patch = roi_mask[y0:y1, x0:x1]

        # Clip to physiological range
        if cfg.clip_range:
            roi_patch = np.clip(roi_patch, cfg.clip_range[0], cfg.clip_range[1])

        # Set non-ROI pixels to NaN, then quantize
        roi_patch[~mask_patch] = np.nan
        valid = roi_patch[mask_patch]
        if len(valid) < 4:
            return {}

        # Quantize to gray levels (Fixed Bin Number)
        vmin, vmax = np.nanmin(valid), np.nanmax(valid)
        if vmax <= vmin:
            return {}

        quantized = np.zeros_like(roi_patch, dtype=np.uint8)
        quantized[mask_patch] = np.clip(
            ((valid - vmin) / (vmax - vmin) * (cfg.gray_levels - 1)).astype(np.uint8),
            0,
            cfg.gray_levels - 1,
        )
        # Set non-ROI to 0 — we'll handle this by only computing on valid pairs
        quantized[~mask_patch] = 0

        # Compute GLCM
        glcm = graycomatrix(
            quantized,
            distances=cfg.distances,
            angles=cfg.angles,
            levels=cfg.gray_levels,
            symmetric=True,
            normed=True,
        )

        # Zero out row/col 0 if non-ROI pixels got mapped there
        # (non-ROI pixels are set to 0 which pollutes the GLCM)
        if not mask_patch.all():
            glcm[0, :, :, :] = 0
            glcm[:, 0, :, :] = 0
            # Re-normalize
            total = glcm.sum(axis=(0, 1), keepdims=True)
            total[total == 0] = 1
            glcm = glcm / total

        results = {}

        # scikit-image features (averaged over angles)
        for feat in cfg.features:
            if feat in SKIMAGE_FEATURES:
                vals = graycoprops(glcm, feat)  # (n_distances, n_angles)
                results[feat] = float(vals.mean())

        # Manual features
        if "variance" in cfg.features:
            results["variance"] = self._glcm_variance(glcm)

        if "entropy" in cfg.features:
            results["entropy"] = self._glcm_entropy(glcm)

        return results

    @staticmethod
    def _glcm_variance(glcm: np.ndarray) -> float:
        """GLCM variance: sum_ij (i - mu)^2 * P(i,j), averaged over angles."""
        n_dist, n_ang = glcm.shape[2], glcm.shape[3]
        vals = []
        for d in range(n_dist):
            for a in range(n_ang):
                p = glcm[:, :, d, a]
                levels = np.arange(p.shape[0])
                mu = np.sum(levels[:, None] * p)
                var = np.sum((levels[:, None] - mu) ** 2 * p)
                vals.append(var)
        return float(np.mean(vals))

    @staticmethod
    def _glcm_entropy(glcm: np.ndarray) -> float:
        """GLCM entropy: -sum_ij P(i,j) * log2(P(i,j)), averaged over angles."""
        n_dist, n_ang = glcm.shape[2], glcm.shape[3]
        vals = []
        for d in range(n_dist):
            for a in range(n_ang):
                p = glcm[:, :, d, a]
                p_nonzero = p[p > 0]
                entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
                vals.append(entropy)
        return float(np.mean(vals))

    def _run_pyradiomics(self, param_map: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
        """Run texture analysis using pyradiomics."""
        try:
            from radiomics import featureextractor
        except ImportError:
            raise ImportError("pyradiomics not installed. Run: uv add pyradiomics")

        import SimpleITK as sitk

        cfg = self.config
        rows = []
        rois = np.unique(mask)
        rois = rois[rois > 0]

        # PyRadiomics config
        settings = {
            "binCount": cfg.gray_levels,
            "symmetricalGLCM": True,
            "distances": cfg.distances,
            "force2D": True,
            "force2Ddimension": 2,  # slice along z
            "normalize": False,
        }
        if cfg.clip_range:
            settings["resegmentRange"] = list(cfg.clip_range)

        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName("firstorder")
        extractor.enableFeatureClassByName("glcm")

        for roi in rois:
            # Create binary mask for this ROI
            roi_mask_bin = (mask == roi).astype(np.int16)

            # Convert to SimpleITK
            img_sitk = sitk.GetImageFromArray(param_map.transpose(2, 0, 1).astype(np.float64))
            mask_sitk = sitk.GetImageFromArray(roi_mask_bin.transpose(2, 0, 1))

            try:
                result = extractor.execute(img_sitk, mask_sitk)
            except Exception:
                continue

            for key, val in result.items():
                if key.startswith("original_"):
                    feat_name = key.replace("original_firstorder_", "fo_").replace(
                        "original_glcm_", "glcm_"
                    )
                    rows.append(
                        {
                            "roi": int(roi),
                            "slice": -1,  # pyradiomics aggregates over volume
                            "feature": feat_name,
                            "value": float(val),
                        }
                    )

        return pd.DataFrame(rows)

    @staticmethod
    def plot_violin_features(
        df: pd.DataFrame,
        features: list[str] | None = None,
        figsize: tuple[int, int] = (14, 8),
        save_path: Path | str | None = None,
    ) -> plt.Figure:
        """Plot violin plots for texture features in 2×3 subplot grid.

        Args:
            df: Aggregated DataFrame with columns: roi, feature, mean, std, n_slices
            features: List of feature names to plot. Default: ['mean'] + 5 GLCM features
            figsize: Figure size (width, height)
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure object
        """
        if features is None:
            features = ["mean", "contrast", "homogeneity", "energy", "variance", "entropy"]

        if len(features) != 6:
            raise ValueError(f"Expected 6 features for 2×3 layout, got {len(features)}")

        # Ensure 'feature' column exists and lowercase for comparison
        if "feature" not in df.columns:
            raise ValueError("DataFrame must have 'feature' column")

        df_lower = df.copy()
        df_lower["feature"] = df_lower["feature"].str.lower()
        features_lower = [f.lower() for f in features]

        fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
        axes = axes.flatten()

        for idx, feat in enumerate(features_lower):
            ax = axes[idx]
            subset = df_lower[df_lower["feature"] == feat]

            if subset.empty:
                ax.text(0.5, 0.5, f"No data for '{feat}'", ha="center", va="center")
                ax.set_title(feat.capitalize())
                continue

            # Violin plot grouped by ROI
            roi_data = [group["mean"].values for _, group in subset.groupby("roi")]
            roi_labels = [str(int(roi)) for roi, _ in subset.groupby("roi")]

            parts = ax.violinplot(roi_data, positions=range(len(roi_data)), widths=0.7)

            # Style violin plot
            for pc in parts["bodies"]:
                pc.set_facecolor("C0")
                pc.set_alpha(0.7)

            ax.set_xticks(range(len(roi_labels)))
            ax.set_xticklabels(roi_labels)
            ax.set_xlabel("ROI")
            ax.set_ylabel("Mean Value")
            ax.set_title(feat.capitalize())
            ax.grid(True, alpha=0.3, axis="y")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
