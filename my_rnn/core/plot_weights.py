import matplotlib.pyplot as plt
import torch


def plot_weight_distribution(weights, title='Weight Distribution', filename='weight_distribution.png'):
    """
    Plot the distribution of the given weights and save it to a file.

    Args:
        weights (torch.Tensor): The weights to plot.
        title (str): The title of the plot.
        filename (str): The filename to save the plot.
    """
    weights = weights.detach().cpu().numpy().flatten()
    plt.hist(weights, bins=50, alpha=0.75, color='blue')
    plt.title(title)
    plt.xlabel('Weight value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show() 