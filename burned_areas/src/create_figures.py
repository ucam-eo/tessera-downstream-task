"""
Module for creating burned area figures from intermediate results.

This module provides functions to load pre-computed UMAP and PCA results
and create publication-ready figures for burned area analysis.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

import visualize

# Local constants for this project
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"


def load_umap_results(
    results_path: Union[str, Path], filename: str = "umap_results_burn_scar_1.npz"
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load UMAP results from saved .npz file.

    Parameters:
    -----------
    results_path : str or Path
        Path to the directory containing the results files
    filename : str, optional
        Name of the UMAP results file (default: "umap_results_burn_scar_1.npz")

    Returns:
    --------
    umap_coords : np.ndarray
        2D array of UMAP coordinates (n_samples, 2)
    labels : np.ndarray
        1D array of severity labels (n_samples,)
    """
    filepath = Path(results_path) / filename
    data = np.load(filepath)

    umap_coords = data["umap_coords"]
    labels = data["labels"]

    return umap_coords, labels


def get_default_burn_severity_config() -> Dict:
    """
    Get default configuration for burn severity visualization.

    Returns:
    --------
    config : dict
        Dictionary containing valid_values, label_map, color_map, and manual_annotations
    """
    return {
        "valid_values": [0, 2, 3, 4],
        "label_map": {0: "Unburned", 2: "Low", 3: "Moderate", 4: "High"},
        "color_map": {
            0: "#155263",  # Unburned -> Dark Blue
            2: "#FFC93C",  # Low -> Yellow
            3: "#FF6F3C",  # Moderate -> Orange
            4: "#810000",  # High -> Dark Red
        },
        "manual_annotations": [
            (-6, -1.5, "June Fire"),
            (0, -5.5, "August Fire"),
            (6, 7, "Unburned"),
        ],
    }


def create_umap_figure(
    results_path: Union[str, Path],
    umap_filename: str = "umap_results_burn_scar_1.npz",
    config: Optional[Dict] = None,
    point_size: int = 10,
    title: str = "",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    legend: bool = True,
) -> None:
    """
    Create and optionally save a UMAP figure from intermediate results.

    Parameters:
    -----------
    results_path : str or Path
        Path to the directory containing the results files
    umap_filename : str, optional
        Name of the UMAP results file
    config : dict, optional
        Configuration dictionary with valid_values, label_map, color_map,
        and manual_annotations.
        If None, uses default burn severity configuration.
    point_size : int, optional
        Size of scatter plot points (default: 10)
    title : str, optional
        Title for the plot (default: "")
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (8, 6))
    save_path : str or Path, optional
        Path to save the figure. If None, only displays the plot.
    dpi : int, optional
        Resolution for saved figure (default: 300)
    legend : bool, optional
        Whether to show the legend (default: True)
    """
    # Load UMAP results
    umap_coords, labels = load_umap_results(results_path, umap_filename)

    # Use default config if none provided
    if config is None:
        config = get_default_burn_severity_config()

    # Create the plot
    visualize.plot_umap_selected_labels(
        umap_coords,
        labels,
        valid_values=config["valid_values"],
        label_map=config["label_map"],
        color_map=config["color_map"],
        point_size=point_size,
        title=title,
        figsize=figsize,
        manual_annotations=config["manual_annotations"],
        legend=legend,
    )

    # Save if path provided
    if save_path is not None:
        import matplotlib.pyplot as plt

        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")


def create_label_efficiency_figure(
    results_path: Union[str, Path] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> None:
    """
    Create and optionally save a label efficiency comparison figure from saved results.

    Parameters
    ----------
    results_path : str or Path, optional
        Path to the directory containing the training results. If None, uses DATA_DIR.
    save_path : str or Path, optional
        Path to save the figure. If None, only displays the plot.
    dpi : int
        Resolution for saved figure (default: 300)
    """
    if results_path is None:
        results_path = DATA_DIR
    input_path = Path(results_path)

    # Load training results from npz
    data = np.load(input_path / "training_results.npz")

    ratios = data["ratios"].tolist()
    sample_counts = data["sample_counts"].tolist()

    # Reconstruct scores dictionary
    scores = {}
    for key in data.keys():
        if key.startswith("scores_"):
            approach_name = (
                key.replace("scores_", "").replace("_", " ").replace(" ", "-")
            )
            # Restore original names
            if "tessera" in approach_name.lower():
                approach_name = "TESSERA"
            elif "gse" in approach_name.lower():
                approach_name = "GSE"
            elif (
                "task" in approach_name.lower() and "specific" in approach_name.lower()
            ):
                approach_name = "Task-Specific Composite"

            scores[approach_name] = data[key].tolist()

    print(f"Training results loaded from {input_path / 'training_results.npz'}")

    # Create the plot
    visualize.plot_label_efficiency_comparison(ratios, scores, sample_counts)

    # Save if path provided
    if save_path is not None:
        import matplotlib.pyplot as plt

        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Label efficiency figure saved to: {save_path}")


def create_label_efficiency_figure_zoomin(
    results_path: Union[str, Path] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> None:
    """
    Create and optionally save a zoomed-in label efficiency comparison figure focusing on
    TESSERA and Task-Specific Composite from 1% to 100% training data.

    Parameters
    ----------
    results_path : str or Path, optional
        Path to the directory containing the training results. If None, uses DATA_DIR.
    save_path : str or Path, optional
        Path to save the figure. If None, only displays the plot.
    dpi : int
        Resolution for saved figure (default: 300)
    """
    if results_path is None:
        results_path = DATA_DIR
    input_path = Path(results_path)

    # Load training results from npz
    data = np.load(input_path / "training_results.npz")

    ratios = data["ratios"].tolist()
    sample_counts = data["sample_counts"].tolist()

    # Reconstruct scores dictionary
    scores = {}
    for key in data.keys():
        if key.startswith("scores_"):
            approach_name = (
                key.replace("scores_", "").replace("_", " ").replace(" ", "-")
            )
            # Restore original names
            if "tessera" in approach_name.lower():
                approach_name = "TESSERA"
            elif "gse" in approach_name.lower():
                approach_name = "GSE"
            elif (
                "task" in approach_name.lower() and "specific" in approach_name.lower()
            ):
                approach_name = "Task-Specific Composite"

            scores[approach_name] = data[key].tolist()

    print(f"Training results loaded from {input_path / 'training_results.npz'}")

    # Create the zoomed-in plot
    visualize.plot_label_efficiency_comparison_zoomin(ratios, scores, sample_counts)

    # Save if path provided
    if save_path is not None:
        import matplotlib.pyplot as plt

        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Label efficiency zoomin figure saved to: {save_path}")


def create_comparison_maps_figure(
    results_path: Union[str, Path] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> None:
    """
    Create and optionally save a comparison maps figure from saved results.

    Parameters
    ----------
    results_path : str or Path, optional
        Path to the directory containing the comparison map data. If None, uses DATA_DIR.
    save_path : str or Path, optional
        Path to save the figure. If None, only displays the plot.
    dpi : int
        Resolution for saved figure (default: 300)
    """
    if results_path is None:
        results_path = DATA_DIR
    input_path = Path(results_path)

    # Load comparison map data from npz
    data = np.load(input_path / "comparison_maps.npz")

    gt_A = data["gt_A"]
    pred_A = data["pred_A"]
    gt_B = data["gt_B"]
    pred_B = data["pred_B"]

    print(f"Comparison maps loaded from {input_path / 'comparison_maps.npz'}")

    # Create the plot
    visualize.plot_comparison_maps(gt_A, pred_A, gt_B, pred_B)

    # Save if path provided
    if save_path is not None:
        import matplotlib.pyplot as plt

        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Comparison maps figure saved to: {save_path}")


def create_fp_fn_analysis_figure(
    results_path: Union[str, Path] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> None:
    """
    Create and optionally save a FP/FN analysis figure from saved results.

    Parameters
    ----------
    results_path : str or Path, optional
        Path to the directory containing the comparison map data. If None, uses DATA_DIR.
    save_path : str or Path, optional
        Path to save the figure. If None, only displays the plot.
    dpi : int
        Resolution for saved figure (default: 300)
    """
    if results_path is None:
        results_path = DATA_DIR
    input_path = Path(results_path)

    # Load comparison map data from npz
    data = np.load(input_path / "comparison_maps.npz")

    gt_A = data["gt_A"]
    pred_A = data["pred_A"]
    gt_B = data["gt_B"]
    pred_B = data["pred_B"]

    print(f"Comparison maps loaded from {input_path / 'comparison_maps.npz'}")

    # Create the FP/FN analysis plot
    visualize.plot_fp_fn_analysis(gt_A, pred_A, gt_B, pred_B)

    # Save if path provided
    if save_path is not None:
        import matplotlib.pyplot as plt

        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"FP/FN analysis figure saved to: {save_path}")


if __name__ == "__main__":
    # Create figures directory if it doesn't exist
    FIGURES_DIR.mkdir(exist_ok=True)

    # Create UMAP figure - Image #1: No text annotations and no legend
    config = get_default_burn_severity_config()
    config["manual_annotations"] = []  # Remove all text annotations

    create_umap_figure(
        results_path=DATA_DIR,
        save_path=FIGURES_DIR / "umap_burned_areas.png",
        point_size=15,  # 1.5x of 10
        figsize=(10, 8),
        config=config,
        legend=False,  # Disable legend
    )

    # Create label efficiency figure
    create_label_efficiency_figure(
        save_path=FIGURES_DIR / "label_efficiency_comparison.png",
    )

    # Create label efficiency zoomin figure
    create_label_efficiency_figure_zoomin(
        save_path=FIGURES_DIR / "label_efficiency_comparison_zoomin.png",
    )

    # Create comparison maps figure
    create_comparison_maps_figure(
        save_path=FIGURES_DIR / "comparison_maps.png",
    )

    # Create FP/FN analysis figure
    create_fp_fn_analysis_figure(
        save_path=FIGURES_DIR / "fp_fn_analysis.png",
    )
