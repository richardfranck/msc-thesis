import matplotlib.pyplot as plt
import numpy as np

# Plot layout and resolution configurations
N_DENSE = 200
N_COLS = 3

def _add_true_curve(axis, dataset, i, times):
    """Draw noise-free, ground truth data-generating curve onto an axis.

    Args:
        axis (plt.Axes): The target matplotlib subplot axis to receive the plot layer.
        dataset (SimulatedDataset): A simulated dataset instance.
        i (int): Index of a sample in the dataset.
        times: A sorted np.ndarray of the observation times for individual i

    Returns:
        None
    """
    y_true = dataset.true_curves_at(times, indices=[i])[0]
    axis.plot(times, y_true, lw=2, color="royalblue", label="True Curve")


def _add_noisy_observations(axis, dataset, i):
    """Plot the observed noisy outcome data points onto an axis.

    Args:
        axis (plt.Axes): The target matplotlib subplot axis to receive the plot layer.
        dataset (SimulatedDataset): A SimulatedDataset instance.
        i (int): Index of a sample in the dataset.

    Returns:
        None
    """
    axis.scatter(
            dataset.times[i],
            dataset.Y_noisy[i],
            s=24, # Size
            alpha=0.7, # Transparency
            color="crimson",
            label="Noisy Obs",
            zorder=3, # Pulled to the front.
        )


def _add_estimated_curve(axis, inference_engine, dataset, i, times):
    """Plot the model's predicted continuous mean trajectory onto an axis.

    Args:
        axis (plt.Axes): The target matplotlib subplot axis to receive the plot layer.
        inference_engine (InferenceEngine): A populated inference tracking model engine.
        dataset (SimulatedDataset): The source evaluation dataset container.
        i (int): Index of a sample in the dataset.
        times (np.ndarray): 1-D sequence of time points in [0, T] at which 
            to evaluate the mean trajectory. This is an evaluation grid used for plotting.

    Returns:
        None
    """
    # Step 1: Get a dict with the coavariate values for individual i
    x_profile = {name: dataset.X[name][i] for name in dataset.X.keys()}

    # Step 2: Predict the outcome valu at times and convert the tensor to an np.array
    y_pred_tensor = inference_engine.predict_trajectory_values(x_profile, times)
    y_pred = y_pred_tensor.detach().cpu().numpy()

    # Step 3: Draw the line
    axis.plot(
        times,
        y_pred,
        lw=2,  # Line width
        linestyle="--",
        color="darkorange",
        label="Model Fit",
    )

def plot_individual_diagnostic(
    dataset,
    indices,
    inference_engine,
    show_true=True,
    show_noisy=True,
    show_estimated=True,
):
    """Orchestrate atomic plot components into a grid layout for individual analysis.

    Args:
        dataset (SimulatedDataset): The simulated source dataset.
        indices (iterable): Integer indices of specific subjects to plot.
        inference_engine (InferenceEngine): A trained model engine.
        show_true (bool, optional): Toggles drawing the generator line.
        show_noisy (bool, optional): Toggles scattering observed dataset points.
        show_estimated (bool, optional): Toggles drawing the model's prediction line..

    Returns:
        plt.Figure: The complete multi-panel diagnostic figure.
    """
    # Step 1: Materialize indices and establish target grid dimensions
    indices = list(indices)
    k = len(indices)
    ncols = min(N_COLS, k)
    nrows = int(np.ceil(k / ncols))

    # Step 2: Initialize canvas and continuous tracking timeline
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
    )
    times = np.linspace(0.0, dataset.T, N_DENSE)

    # Step 3: Iterate over subjects and systematically layer active visual components
    for ax, i in zip(axes.flat, indices):

        if show_true:
            _add_true_curve(ax, dataset, i, times)

        if show_noisy:
            _add_noisy_observations(ax, dataset, i)

        if show_estimated and inference_engine is not None:
            _add_estimated_curve(ax, inference_engine, dataset, i, times)

        # Step 4: Apply clean labels and individual identification headers
        ax.set_title(f"Individual i={i}", fontsize=9)
        ax.set_xlabel("t")
        ax.set_ylabel("y")

    # Step 5: Shut off unneeded canvas regions and anchor uniform formatting
    for ax in axes.flat[k:]:
        ax.axis("off")

    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()

    return fig



def plot_sweep(sweep_results, xlabel, logx=False):
    """Line plot of shape-extraction accuracy against a swept configuration axis.

    This function permits plotting the results from e.g. evaluation.sweep_knots
    or an evaluation.sweep_observations.

    Args:
        sweep_results: List of (x_value, accuracy) tuples from an evaluator sweep.
        xlabel (str): Label for the horizontal swept axis (e.g., "Number of Knots").
        logx (bool): If True, plots the x-axis on a logarithmic scale.

    Returns:
        plt.Figure: The generated matplotlib figure containing the trend line.
    """
    # Step 1: Unpack the sweep tuples using zip
    x_values, accuracies = zip(*sweep_results)
    
    # Step 2: Build the layout canvas
    fig, ax = plt.subplots(figsize=(6, 4)) # 6 inches width x 4 inches tall
    
    # Step 3: Add the trend lines and data markers
    ax.plot(
        x_values, 
        accuracies, 
        marker="o", # place dots on datapoints
        color="#1f77b4",  # blue
        linewidth=2, 
        markersize=6
    )
    
    # Step 4: Scale adjustments
    if logx:
        ax.set_xscale("log")
        
    # Step 5: Labels, Titles, and Bounds
    ax.set_xlabel(xlabel, fontsize=10, fontweight="medium")
    ax.set_ylabel("Exact Sequence Match Share", fontsize=10, fontweight="medium")
    
    # Since accuracy is a percentage share, lock bounds for visual consistency
    ax.set_ylim(-0.05, 1.05)
    
    fig.tight_layout()
    return fig