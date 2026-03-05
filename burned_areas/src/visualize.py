"""
Visualization functions for burned area analysis.

This module contains plotting functions for UMAP and other dimensionality reduction
visualizations used in burned area analysis.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb, ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D


def plot_umap_selected_labels(
    umap_2d: np.ndarray,
    labels: np.ndarray,
    valid_values: Optional[List] = None,
    label_map: Optional[Dict] = None,
    color_map: Optional[Dict] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    point_size: int = 6,
    alpha: float = 0.4,
    legend_loc: str = "upper right",
    manual_annotations: Optional[List[Tuple[float, float, str]]] = None,
    legend: bool = True,
) -> None:
    """
    Plot precomputed UMAP embeddings with filtered classes and custom label/color maps.

    Parameters:
    -----------
    umap_2d : np.ndarray
        (N, 2) array of UMAP-reduced embeddings
    labels : np.ndarray
        (N,) array of class labels
    valid_values : list, optional
        List of label values to include (others excluded)
    label_map : dict, optional
        Dictionary mapping label values to display names
    color_map : dict, optional
        Dictionary mapping label values to RGB or hex colors
    title : str, optional
        Optional plot title
    figsize : tuple, optional
        Plot size (width, height) in inches (default: (8, 6))
    point_size : int, optional
        Scatter point size (default: 6)
    alpha : float, optional
        Point transparency (default: 0.4)
    legend_loc : str, optional
        Where to place the legend (default: "upper right")
    manual_annotations : list, optional
        List of (x, y, text) annotations to add to the plot
    legend : bool, optional
        Whether to show the legend (default: True)
    """
    df = pd.DataFrame(umap_2d, columns=["UMAP 1", "UMAP 2"])
    df["label"] = labels

    if valid_values is not None:
        df = df[df["label"].isin(valid_values)]

    unique_labels = sorted(df["label"].unique())

    # Set color palette
    if color_map:
        # Convert any hex strings to RGB
        color_map = {
            val: (
                to_rgb(color_map[val])
                if isinstance(color_map[val], str)
                else color_map[val]
            )
            for val in unique_labels
            if val in color_map
        }
    else:
        palette = sns.color_palette("colorblind", len(unique_labels))
        color_map = dict(zip(unique_labels, palette))

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.scatterplot(
        data=df,
        x="UMAP 1",
        y="UMAP 2",
        hue="label",
        palette=color_map,
        s=point_size,
        alpha=0.7,  # Set transparency to 0.7
        legend=False,
    )

    # Custom legend
    if legend:
        legend_elements = []
        for label in unique_labels:
            label_str = label_map.get(label, str(label)) if label_map else str(label)
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=label_str,
                    markerfacecolor=color_map[label],
                    markersize=6,
                )
            )
        ax.legend(handles=legend_elements, title="Class", loc=legend_loc, frameon=True)

    # Manual cluster annotations
    if manual_annotations:
        for x, y, text in manual_annotations:
            ax.text(
                x,
                y,
                text,
                fontsize=10,
                weight="bold",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
            )

    # Styling
    ax.set_xlabel("UMAP 1", fontsize=18)
    ax.set_ylabel("UMAP 2", fontsize=18)
    ax.spines[["top", "right"]].set_visible(False)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 18,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "black",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
        }
    )
    plt.grid(False)
    plt.tight_layout()
    plt.title(title or "", fontsize=12)
    plt.show()


def plot_label_efficiency_comparison(ratios, combined_scores_dict, sample_counts):
    """
    Plots a publication-quality comparison of model performance against the
    ratio of training data used, including a second x-axis for absolute sample counts.

    This function creates a dual-axis plot showing both the fraction of training data
    used (bottom x-axis) and the absolute number of training samples (top x-axis).
    It's designed for comparing multiple approaches or models on the same plot.

    Parameters
    ----------
    ratios : list of float
        List of ratios (0.0 to 1.0) representing the fraction of training data used.
        Should be in ascending order.
    combined_scores_dict : dict
        Dictionary where keys are approach/model names (strings) and values are
        lists of F1-scores corresponding to each ratio in `ratios`.
        Example: {"TESSERA": [0.8, 0.85, 0.9], "GSE": [0.75, 0.82, 0.88]}
    sample_counts : list of int
        List of absolute sample counts corresponding to each ratio in `ratios`.
        Should have the same length as `ratios`.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Examples
    --------
    >>> ratios = [0.001, 0.01, 0.1, 1.0]
    >>> sample_counts = [100, 1000, 10000, 100000]
    >>> scores = {
    ...     "TESSERA": [0.8, 0.85, 0.9, 0.95],
    ...     "GSE": [0.75, 0.82, 0.88, 0.93]
    ... }
    >>> plot_label_efficiency_comparison(ratios, scores, sample_counts)
    """

    # --- Helper function for formatting numbers ---
    def format_number(n):
        """Format numbers for display on the top x-axis."""
        if n < 1000:
            return str(n)
        elif n < 10000:
            return f"{n/1000:.1f}k"
        elif n < 1000000:
            return f"{int(n/1000)}k"
        else:
            return f"{n/1000000:.1f}M"

    # --- Helper function for formatting ratios ---
    def format_ratio(r):
        """Format ratio percentages to remove unnecessary trailing zeros."""
        percentage = r * 100
        if percentage == int(percentage):
            return f"{int(percentage)}%"
        elif percentage == round(percentage, 1):
            return f"{percentage:.1f}%"
        elif percentage == round(percentage, 2):
            return f"{percentage:.2f}%"
        else:
            return f"{percentage:.3f}%"

    # --- Plot Styling ---
    plt.rcParams.update({"font.size": 14, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define a color cycle for the lines
    colors = ["#6A2C70", "#F08A5D", "#2D6A4F"]  # Purple for TESSERA, Orange for Composite, Green for GSE
    color_cycle = iter(colors)

    # --- Plotting the Data ---
    for approach_name, scores in combined_scores_dict.items():
        ax.plot(
            ratios,
            scores,
            marker="o",
            linestyle="-",
            label=approach_name,
            markersize=8,
            zorder=10,
            color=next(color_cycle),
            linewidth=2.5,
        )

    # --- Aesthetics ---
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)

    # --- Bottom X-axis (Percentages) ---
    ax.set_xlabel("Fraction of Training Data Used", fontsize=18, labelpad=10)
    ax.set_ylabel("Average F1-Score", fontsize=18, labelpad=10)
    ax.tick_params(axis="x", which="major", labelsize=14, pad=7, width=1.5)
    ax.tick_params(axis="y", which="major", labelsize=14, width=1.5)
    ax.set_xscale("log")
    ax.set_xticks(ratios)
    ax.set_xticklabels([format_ratio(r) for r in ratios], rotation=45, ha="right")

    # Ensure y-axis starts at 0
    ax.set_ylim(0.3, 1.00)

    # --- Top X-axis (Sample Counts) ---
    ax2 = ax.twiny()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_linewidth(1.5)
    ax2.spines["left"].set_linewidth(1.5)
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ratios)
    # Use the helper function to format the labels
    ax2.set_xticklabels(
        [format_number(count) for count in sample_counts], rotation=45, ha="left"
    )
    ax2.set_xlabel("Number of Training Samples", fontsize=18, labelpad=10)
    ax2.tick_params(axis="x", which="major", labelsize=14, pad=7, width=1.5)

    # --- Legend ---
    ax.legend(loc="lower right", fontsize=14, frameon=False)

    plt.tight_layout()
    plt.show()


def plot_label_efficiency_comparison_zoomin(ratios, combined_scores_dict, sample_counts):
    """
    Creates a zoomed-in version of the label efficiency comparison plot focusing on
    TESSERA and Task-Specific Composite performance from 1% to 100% training data.

    Parameters
    ----------
    ratios : list of float
        List of ratios (0.0 to 1.0) representing the fraction of training data used.
    combined_scores_dict : dict
        Dictionary where keys are approach/model names and values are F1-scores.
    sample_counts : list of int
        List of absolute sample counts (not used in this zoomed version).
    """
    # Filter data for 1% to 100% range (ratios >= 0.01)
    zoom_indices = [i for i, r in enumerate(ratios) if r >= 0.01]
    zoom_ratios = [ratios[i] for i in zoom_indices]

    # Extract only TESSERA and Task-Specific Composite data
    target_approaches = ["TESSERA", "Task-Specific Composite"]
    zoom_scores = {}

    for approach_name, scores in combined_scores_dict.items():
        if approach_name in target_approaches:
            zoom_scores[approach_name] = [scores[i] for i in zoom_indices]

    # Calculate y-axis range based on the data
    all_values = []
    for scores in zoom_scores.values():
        all_values.extend(scores)
    y_min = min(all_values) - 0.01
    y_max = max(all_values) + 0.01

    # --- Helper function for formatting ratios ---
    def format_ratio(r):
        """Format ratio percentages to remove unnecessary trailing zeros."""
        percentage = r * 100
        if percentage == int(percentage):
            return f"{int(percentage)}%"
        elif percentage == round(percentage, 1):
            return f"{percentage:.1f}%"
        elif percentage == round(percentage, 2):
            return f"{percentage:.2f}%"
        else:
            return f"{percentage:.3f}%"

    # --- Plot Styling ---
    plt.rcParams.update({"font.size": 14, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(16, 4))  # 4:1 width to height ratio

    # Define colors for TESSERA and Task-Specific Composite
    approach_colors = {
        "TESSERA": "#6A2C70",
        "Task-Specific Composite": "#F08A5D"
    }

    # --- Plotting the Data ---
    for approach_name, scores in zoom_scores.items():
        ax.plot(
            zoom_ratios,
            scores,
            marker="o",
            linestyle="-",
            label=approach_name,
            markersize=16,  # Doubled from 8 to 16
            zorder=10,
            color=approach_colors[approach_name],
            linewidth=2.5,
        )

    # --- Aesthetics ---
    # Show all four spines (frame around the plot)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    # --- X-axis configuration ---
    ax.tick_params(axis="x", which="major", labelsize=28, pad=7, width=1.5)  # Doubled font size
    ax.tick_params(axis="y", which="major", labelsize=28, width=1.5)  # Doubled font size
    ax.set_xscale("log")
    ax.set_xticks(zoom_ratios)
    ax.set_xticklabels([format_ratio(r) for r in zoom_ratios], rotation=45, ha="right")

    # Remove axis labels (only keep ticks and tick labels)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Set fixed y-axis range and custom y-ticks
    ax.set_ylim(0.950, 0.965)
    ax.set_yticks([0.9500, 0.9575, 0.9650])

    # No legend for zoomin version

    plt.tight_layout()
    plt.show()


def plot_comparison_maps(gt_A, pred_A, gt_B, pred_B):
    """
    Plots a 1x4 wide image comparing ground truth vs. prediction for both areas.

    This function creates a side-by-side comparison of ground truth and prediction
    maps for two areas (A and B) in a single row layout. It uses a custom colormap
    to properly display binary classification results with NoData values.

    Parameters
    ----------
    gt_A : np.ndarray
        2D array containing ground truth labels for Area A
    pred_A : np.ndarray
        2D array containing prediction labels for Area A
    gt_B : np.ndarray
        2D array containing ground truth labels for Area B
    pred_B : np.ndarray
        2D array containing prediction labels for Area B

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Examples
    --------
    >>> # Assuming you have ground truth and prediction maps for two areas
    >>> plot_comparison_maps(gt_map_A, pred_map_A, gt_map_B, pred_map_B)
    """
    print("Generating 1x4 comparison map...")

    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    # --- Correct Color Mapping ---
    # Unburned (0) -> Blue, Burned (1) -> Red, NoData/Other (-1) -> Gray
    cmap_dict = {
        0: "#1E93AB",  # Unburned -> Blue
        1: "#E62727",  # Burned -> Red
        -1: "#cccccc",  # NoData/Other -> Gray
    }
    codes = sorted(cmap_dict.keys())
    colors = [cmap_dict[c] for c in codes]
    cmap = ListedColormap(colors)
    boundaries = [c - 0.5 for c in codes] + [codes[-1] + 0.5]
    norm = BoundaryNorm(boundaries, cmap.N)

    # Plot Area A Ground Truth
    axes[0].imshow(gt_A, cmap=cmap, norm=norm)
    axes[0].set_title("Area A: Ground Truth", fontsize=18)
    axes[0].axis("off")

    # Plot Area A Prediction
    axes[1].imshow(pred_A, cmap=cmap, norm=norm)
    axes[1].set_title("Area A: Prediction", fontsize=18)
    axes[1].axis("off")

    # Plot Area B Ground Truth
    axes[2].imshow(gt_B, cmap=cmap, norm=norm)
    axes[2].set_title("Area B: Ground Truth", fontsize=18)
    axes[2].axis("off")

    # Plot Area B Prediction
    axes[3].imshow(pred_B, cmap=cmap, norm=norm)
    axes[3].set_title("Area B: Prediction", fontsize=18)
    axes[3].axis("off")

    # Remove legend
    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.95, bottom=0.1)
    plt.show()


def plot_fp_fn_analysis(gt_A, pred_A, gt_B, pred_B):
    """
    Creates a 2x3 comparison plot showing Ground Truth, Prediction, and FP/FN analysis for two areas.

    This function creates a detailed analysis comparing ground truth vs predictions with
    false positive/false negative breakdown for two areas in a 2x3 layout.

    Parameters
    ----------
    gt_A : np.ndarray
        2D array containing ground truth labels for Area A
    pred_A : np.ndarray
        2D array containing prediction labels for Area A
    gt_B : np.ndarray
        2D array containing ground truth labels for Area B
    pred_B : np.ndarray
        2D array containing prediction labels for Area B

    Returns
    -------
    None
        Displays the plot using matplotlib.
    """
    print("Generating 2x3 FP/FN analysis comparison...")

    fig, axes = plt.subplots(2, 3, figsize=(21, 14))

    # Define color maps for GT/Pred (burned/unburned)
    gt_pred_cmap_dict = {
        0: "#1E93AB",  # Unburned
        1: "#E62727",  # Burned
        -1: "#cccccc", # NoData/Other
    }

    # Define color map for FP/FN analysis
    # TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative
    fp_fn_cmap_dict = {
        0: "#2D4059",  # True Negative (correctly predicted unburned)
        1: "#EA5455",  # False Positive (wrongly predicted burned)
        2: "#F07B3F",  # False Negative (wrongly predicted unburned)
        3: "#FFD460",  # True Positive (correctly predicted burned)
        -1: "#cccccc", # NoData/Other
    }

    def create_colormap_and_norm(cmap_dict):
        codes = sorted(cmap_dict.keys())
        colors = [cmap_dict[c] for c in codes]
        cmap = ListedColormap(colors)
        boundaries = [c - 0.5 for c in codes] + [codes[-1] + 0.5]
        norm = BoundaryNorm(boundaries, cmap.N)
        return cmap, norm

    def calculate_fp_fn_map(gt, pred):
        """Calculate FP/FN analysis map"""
        result = np.full_like(gt, -1)  # Initialize with NoData

        # Create masks for valid data (not -1)
        valid_mask = (gt != -1) & (pred != -1)

        # True Negative: GT=0, Pred=0 (correctly predicted unburned)
        tn_mask = valid_mask & (gt == 0) & (pred == 0)
        result[tn_mask] = 0

        # False Positive: GT=0, Pred=1 (wrongly predicted burned)
        fp_mask = valid_mask & (gt == 0) & (pred == 1)
        result[fp_mask] = 1

        # False Negative: GT=1, Pred=0 (wrongly predicted unburned)
        fn_mask = valid_mask & (gt == 1) & (pred == 0)
        result[fn_mask] = 2

        # True Positive: GT=1, Pred=1 (correctly predicted burned)
        tp_mask = valid_mask & (gt == 1) & (pred == 1)
        result[tp_mask] = 3

        return result

    # Create colormaps
    gt_pred_cmap, gt_pred_norm = create_colormap_and_norm(gt_pred_cmap_dict)
    fp_fn_cmap, fp_fn_norm = create_colormap_and_norm(fp_fn_cmap_dict)

    # Calculate FP/FN maps
    fp_fn_A = calculate_fp_fn_map(gt_A, pred_A)
    fp_fn_B = calculate_fp_fn_map(gt_B, pred_B)

    # Row 1: Area A
    axes[0, 0].imshow(gt_A, cmap=gt_pred_cmap, norm=gt_pred_norm)
    axes[0, 0].set_title("Area A: Ground Truth", fontsize=18)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pred_A, cmap=gt_pred_cmap, norm=gt_pred_norm)
    axes[0, 1].set_title("Area A: Prediction", fontsize=18)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(fp_fn_A, cmap=fp_fn_cmap, norm=fp_fn_norm)
    axes[0, 2].set_title("Area A: FP/FN Analysis", fontsize=18)
    axes[0, 2].axis("off")

    # Row 2: Area B
    axes[1, 0].imshow(gt_B, cmap=gt_pred_cmap, norm=gt_pred_norm)
    axes[1, 0].set_title("Area B: Ground Truth", fontsize=18)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_B, cmap=gt_pred_cmap, norm=gt_pred_norm)
    axes[1, 1].set_title("Area B: Prediction", fontsize=18)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(fp_fn_B, cmap=fp_fn_cmap, norm=fp_fn_norm)
    axes[1, 2].set_title("Area B: FP/FN Analysis", fontsize=18)
    axes[1, 2].axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0.1, top=0.95, bottom=0.05)
    plt.show()
