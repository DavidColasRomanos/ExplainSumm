"""
Data visualization module.

"""

import os
import shap
import scipy.stats
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_metrics(metrics_dicts, titles=None, filename=None):
    """
    Visualize classification metrics.

    Parameters:
    metrics_dicts : list of dict
        A list of dictionaries of various classification metrics.
    titles : list of str, optional
        A list of titles for the subplots.
    filename : str, optional
        Path to save the plot as an image file.
    """

    num_metrics = len(metrics_dicts)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 6))

    # Ensure that axes is always a list, even when there's only one subplot
    if num_metrics == 1:
        axes = [axes]

    # If titles are provided, ensure there's a title for each subplot
    if titles and len(titles) != num_metrics:
        raise ValueError("Number of titles must match the number of metrics_dicts")

    for i, metrics_dict in enumerate(metrics_dicts):
        ax = axes[i]

        # Print basic metrics
        print(f"{titles[i]}:")
        for key, value in metrics_dict.items():
            if key != "Confusion Matrix" and key != "Classification Report":
                print(f"{key}: {round(value * 100, 2)} %")
        print("\n")

        # Convert the confusion matrix to percentage
        confusion_matrix_percentage = (
            metrics_dict["Confusion Matrix"]
            / metrics_dict["Confusion Matrix"].sum()
            * 100
        )

        # Confusion matrix
        sns.heatmap(
            confusion_matrix_percentage,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=True,
            annot_kws={"size": 14},
            ax=ax,
        )
        # Set title from titles list or use default
        title = (
            f"{titles[i]} - Confusion Matrix (%)" if titles else "Confusion Matrix (%)"
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()

    if filename:
        # Check if the directory exists, if not, create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the figure
        plt.savefig(filename + ".png", bbox_inches="tight")

    plt.show()


def plot_fraction_preferred_to_ref(df: pd.DataFrame, filename: str):
    """
    Creates a line plot to visualize the 'Fraction preferred to ref' for various 
    models given their parameters. The plot includes a reference line at y=0.5 
    indicating the 'Reference Summaries'.
    
    Parameters:
    df : pd.DataFrame
        Fraction preferrend to renference DataFrame.
    filename : str
        The filename to save the plot, including path.             
        
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

    if filename:
        # Check if the directory exists, if not, create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the figure
        plt.savefig(filename + ".png", bbox_inches="tight")

    plt.show()


def plot_metrics_by_algorithm_and_complexity(eval_summ_metrics, metrics, filename):
    """
    Creates a figure with line plots to visualize each of the given metrics for 
    various summarization algorithms and complexities specified by the number of 
    parameters of the model. The plots are arranged in a grid with 3 columns.

    Parameters:
    eval_summ_metrics : EvalSummMetrics
        EvalSummMetrics instance containing calculated metrics.
    metrics : list
        List of metrics to be plotted.
    filename : str
        The filename to save the plot, including path.        

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

    if filename:
        # Check if the directory exists, if not, create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the figure
        plt.savefig(filename + ".png", bbox_inches="tight")

    plt.show()


def plot_metrics_by_fraction_preferred_to_ref(eval_summ_metrics, metrics, filename):
    """
    Creates a figure with line plots to visualize each of the given metrics for 
    various summarization algorithms and complexities specified by the fraction 
    preferred to ref. The plots are arranged in a grid with 3 columns.

    Parameters:
    eval_summ_metrics : EvalSummMetrics
        EvalSummMetrics instance containing calculated metrics.
    metrics : list
        List of metrics to be plotted.
    filename : str
        The filename to save the plot, including path.             

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
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2})

    # Plot each metric
    for ax, metric in zip(axs, metrics):
        # Line plot
        sns.lineplot(
            data=eval_summ_metrics.metrics_df,
            x="Fraction preferred to ref",
            y=metric,
            color="black",
            linestyle=":",
            marker="o",
            markersize=7.5,
            markers=True,
            dashes=True,
            ax=ax,
        )

        # Calculate correlation
        corr, _ = scipy.stats.pearsonr(
            eval_summ_metrics.metrics_df["Fraction preferred to ref"],
            eval_summ_metrics.metrics_df[metric],
        )

        # Set title, labels
        ax.set_title(f"Correlation: {corr:.2f}")
        ax.set_ylabel(metric)
        ax.set_xlabel("Fraction preferred to ref")

    # If there are fewer metrics than subplots, remove the extra subplots
    for ax in axs[n_metrics:]:
        ax.remove()

    plt.tight_layout()

    if filename:
        # Check if the directory exists, if not, create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the figure
        plt.savefig(filename + ".png", bbox_inches="tight")

    plt.show()


def visualize_correlation(
    df_features, df_targets, target_variable, ax=None, filename=None
):
    """
    Visualize correlation between features and target.

    Parameters:
    df_features : DataFrame
        A DataFrame of feature variables.
    df_targets : DataFrame
        A DataFrame of target variables.
    target_variable : str
        The name of the target variable for correlation calculation.
    ax : matplotlib.axes.Axes
        The axes on which to draw the plot.  
    filename : str
        The filename to save the plot, including path.              
    """

    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})

    # Combine feature and target DataFrames
    df = pd.concat([df_features.astype(float), df_targets.astype(float)], axis=1)

    # Calculate correlation matrix
    correlation_matrix = df.corr()

    # Isolate target-feature correlations and transpose
    target_feature_corr = correlation_matrix.loc[
        df_targets.columns, df_features.columns
    ].T

    # Compute the correlation for each feature with target variable
    overall_corr = target_feature_corr[target_variable]

    # Sort the features based on these correlations
    sorted_features = overall_corr.sort_values(ascending=False).index

    # Apply the sorted features to the target-feature correlation matrix
    sorted_target_feature_corr = target_feature_corr.loc[sorted_features]

    # Create a figure if ax is None
    if ax is None:
        width_size = 2.25 * df_targets.shape[1]
        fig, ax = plt.subplots(figsize=(width_size, 6))  # Change the size as necessary

    sns.heatmap(
        sorted_target_feature_corr,
        annot=True,
        cmap="vlag",
        xticklabels=sorted_target_feature_corr.columns.values,
        yticklabels=sorted_target_feature_corr.index.values,
        ax=ax,
    )

    # Make the target variable label bold on the x-axis
    labels = ax.get_xticklabels()  # get x labels
    for label in labels:
        if label.get_text() == target_variable:
            label.set_weight("bold")  # set the target variable label to bold

    if filename:
        # Check if the directory exists, if not, create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the figure
        plt.savefig(filename + ".png", bbox_inches="tight")


def visualize_distributions(df, filename):
    """
    Visualize the distributions of the target variables.

    Parameters:
    df : DataFrame
        A DataFrame of target variables.
    filename : str
        The filename to save the plot, including path.          

    Returns:
    Plot DataFrame distributions.
    """

    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})

    # Plotting the distributions of the variables
    plt.figure(figsize=(15, 10))

    for i, column in enumerate(df.columns, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[column], bins=6, kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

    plt.tight_layout()

    if filename:
        # Check if the directory exists, if not, create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the figure
        plt.savefig(filename + ".png", bbox_inches="tight")

    plt.show()


def visualize_shap_values(shap_values_list, X_test_list, titles, filename=None):
    """
    Plot summary plots for multiple SHAP values.

    Parameters:
    shap_values_list : list of array-like
        List of SHAP values for the instances to be plotted.
    X_test_list : list of DataFrame
        List of corresponding test data for the SHAP values.
    titles : list of str
        Titles for each of the subplots.
    filename : str, optional
        Path to save the plot as an image file.
    """

    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.5})

    if len(shap_values_list) > 1:

        # Create a figure
        plt.figure(figsize=(29, 12))

        # Loop through the shap_values_list and X_test_list and plot
        for i, (shap_values, X_test, title) in enumerate(
            zip(shap_values_list, X_test_list, titles)
        ):
            plt.subplot(1, len(shap_values_list), i + 1)
            shap.summary_plot(shap_values, X_test, plot_size=None, show=False)
            plt.title(title)

        plt.tight_layout()

    else:
        shap.plots.beeswarm(
            shap_values_list[0], max_display=20, show=False, plot_size=(13, 12)
        )
        plt.title(titles[0])

    if filename:
        # Check if the directory exists, if not, create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the figure
        plt.savefig(filename + ".png", bbox_inches="tight")

    plt.show()


def visualize_shap_values_by_quality(models, labels=None, filename=None):
    """
    Visualize SHAP values by quality levels.

    Parameters:
    models : list
        List of objects that have a dedicated attribute containing SHAP values
        and an X_test attribute containing feature names.
    labels : list of str, optional
        Labels for each set of SHAP values. Default is ["low_quality", "mid_quality", "high_quality"].
    filename : str, optional
        Path to save the plot as an image file.

    Returns:
    shap_df : DataFrame
        A DataFrame containing the mean absolute SHAP values for each feature 
        across the provided models/quality levels.
    """

    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})

    if labels is None:
        labels = ["low_quality", "mid_quality", "high_quality"]

    shap_dfs = []
    for model, label in zip(models, labels):
        shap_values = (
            pd.DataFrame(model.shap_values.values, columns=model.X_test.columns)
            .abs()
            .mean()
            .rename(label)
        )
        shap_dfs.append(shap_values)

    shap_df = pd.concat(shap_dfs, axis=1)

    shap_df["mean"] = shap_df.mean(axis=1)
    shap_df.sort_values("mean", inplace=True)
    shap_df.drop("mean", axis=1, inplace=True)

    # Plotting the mean abs shap values
    shap_df.tail(20).plot(kind="barh", stacked=False, figsize=(15, 12))

    plt.title("SHAP bar plot by Quality Level")
    plt.xlabel("mean(|SHAP value|)")
    plt.ylabel("Metric")
    plt.xticks(rotation=90)
    plt.tight_layout()

    if filename:
        # Check if the directory exists, if not, create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the figure
        plt.savefig(filename + ".png", bbox_inches="tight")

    plt.show()

    return shap_df


def plot_shap_dependence(metric, models, labels=None, filename=None):
    """
    Plot SHAP dependence plots for a given metric across multiple models.

    Parameters:
    metric : str
        The metric for which SHAP dependence plot should be drawn.
    models : list
        List of objects that have a dedicated attribute containing SHAP values
        and an X_test attribute containing feature names.
    labels : list of str, optional
        Labels for each set of SHAP values. Default is ["low_quality", "mid_quality", "high_quality"].
    filename : str, optional
        Path to save the plot as an image file.
    """

    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.5})

    if labels is None:
        labels = ["low_quality", "mid_quality", "high_quality"]

    fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(22, 7))

    for model, ax, label in zip(models, axes, labels):
        shap.dependence_plot(
            metric,
            model.shap_values.values,
            model.X_test,
            ax=ax,
            interaction_index=metric,
            xmin="percentile(1)",
            xmax="percentile(99)",
            show=False,
            title=label,
        )

    plt.tight_layout()

    if filename:
        # Check if the directory exists, if not, create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the figure
        plt.savefig(filename + ".png", bbox_inches="tight")

    plt.show()

