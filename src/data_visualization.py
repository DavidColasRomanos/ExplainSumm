"""
Data visualization module.

"""

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_metrics(metrics_dict):
    """
    Visualize classification metrics.

    Parameters:
    metrics_dict : dict
        A dictionary of various classification metrics.
    """

    # Print basic metrics
    for key, value in metrics_dict.items():
        if key != "Confusion Matrix" and key != "Classification Report":
            print(f"{key}: {round(value * 100, 2)} %")

    # Confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        metrics_dict["Confusion Matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        annot_kws={"size": 14},
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def plot_fraction_preferred_to_ref(df: pd.DataFrame):
    """
    Creates a line plot to visualize the 'Fraction preferred to ref' for various 
    models given their parameters. The plot includes a reference line at y=0.5 
    indicating the 'Reference Summaries'.
    
    Parameters:
    df : pd.DataFrame
        Fraction preferrend to renference DataFrame.
        
    Returns:
    None. The function directly creates a plot using matplotlib and seaborn.
    
    """

    plt.figure(figsize=(12, 7))

    sns.set_palette(sns.color_palette("tab10"))
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    sns.lineplot(
        data=df,
        x="Parameters (B)",
        y="Fraction preferred to ref",
        hue="Model",
        linestyle=":",
        marker="o",
        markersize=7.5,
        markers=True,
        dashes=True,
    )

    # Add a horizontal line at y=0.5
    plt.axhline(0.5, color="black", linestyle="-.", linewidth=1)

    # Add a label for this line
    plt.text(
        12,
        0.51,
        "Reference Summaries",
        fontsize=15,
        va="bottom",
        ha="right",
        bbox=dict(facecolor="w", alpha=0.5),
    )

    plt.ylabel("Fraction Preferred to Reference")
    plt.xlabel("Parameters (B)")

    # Position legend below x-axis
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.show()


def plot_metrics_by_algorithm_and_complexity(eval_summ_metrics, metrics):
    """
    Creates a figure with line plots to visualize each of the given metrics for 
    various summarization algorithms and complexities specified by the number of 
    parameters of the model. The plots are arranged in a grid with 3 columns.

    Parameters:
    eval_summ_metrics : EvalSummMetrics
        EvalSummMetrics instance containing calculated metrics.
    metrics : list
        List of metrics to be plotted.

    Returns:
    None. The function directly creates a plot using matplotlib and seaborn.

    """

    # Number of metrics
    n_metrics = len(metrics)

    # Calculate the number of rows for the subplot grid
    n_rows = (n_metrics + 2) // 3  # '+2' ensures we round up

    # Figure and axes
    fig, axs = plt.subplots(n_rows, 3, figsize=(22, 6 * n_rows))

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    # Sets
    sns.set_palette(sns.color_palette("tab10"))
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})

    # Plot each metric
    for ax, metric in zip(axs, metrics):
        # Line plot
        sns.lineplot(
            data=eval_summ_metrics.metrics_df,
            x="Parameters (B)",
            y=metric,
            hue="Model",
            linestyle=":",
            marker="o",
            markersize=7.5,
            markers=True,
            dashes=True,
            ax=ax,
        )

        ax.set_ylabel(metric)
        ax.set_xlabel("Parameters (B)")

    # If there are fewer metrics than subplots, remove the extra subplots
    for ax in axs[n_metrics:]:
        ax.remove()

    plt.tight_layout()
    plt.show()


def plot_metrics_by_fraction_preferred_to_ref(eval_summ_metrics, metrics):
    """
    Creates a figure with line plots to visualize each of the given metrics for 
    various summarization algorithms and complexities specified by the fraction 
    preferred to ref. The plots are arranged in a grid with 3 columns.

    Parameters:
    eval_summ_metrics : EvalSummMetrics
        EvalSummMetrics instance containing calculated metrics.
    metrics : list
        List of metrics to be plotted.

    Returns:
    None. The function directly creates a plot using matplotlib and seaborn.

    """

    # Number of metrics
    n_metrics = len(metrics)

    # Calculate the number of rows for the subplot grid
    n_rows = (n_metrics + 2) // 3  # '+2' ensures we round up

    # Figure and axes
    fig, axs = plt.subplots(n_rows, 3, figsize=(22, 6 * n_rows))

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    # Sets
    sns.set_palette(sns.color_palette("tab10"))
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})

    # Plot each metric
    for ax, metric in zip(axs, metrics):
        # Line plot
        sns.lineplot(
            data=eval_summ_metrics.metrics_df,
            x="Fraction preferred to ref",
            y=metric,
            linestyle=":",
            marker="o",
            markersize=7.5,
            markers=True,
            dashes=True,
            ax=ax,
        )

        ax.set_ylabel(metric)
        ax.set_xlabel("Fraction preferred to ref")

    # If there are fewer metrics than subplots, remove the extra subplots
    for ax in axs[n_metrics:]:
        ax.remove()

    plt.tight_layout()
    plt.show()

