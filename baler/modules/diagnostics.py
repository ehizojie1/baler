import torch
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os
import modules.plotting as plotting


def get_mean_node_activations(input_dict: dict) -> dict:
    output_dict = {}
    for kk in input_dict:
        output_dict_layer = []
        for node in input_dict[kk].T:
            output_dict_layer.append(torch.mean(node).item())
        output_dict[kk] = output_dict_layer
    return output_dict


def dict_to_square_matrix(input_dict: dict) -> np.array:
    """Function changes an input dictionary into a square np.array. Adds NaNs when the dimension of a dict key is less than of the final square matrix.

    Args:
        input_dict (dict)

    Returns:
        square_matrix (np.array)
    """
    means_dict = get_mean_node_activations(input_dict)
    max_number_of_nodes = 0
    number_of_layers = len(input_dict)
    for kk in means_dict:
        if len(means_dict[kk]) > max_number_of_nodes:
            max_number_of_nodes = len(means_dict[kk])
    square_matrix = np.empty((number_of_layers, max_number_of_nodes))
    counter = 0
    for kk in input_dict:
        layer = np.array(means_dict[kk])
        if len(layer) == max_number_of_nodes:
            square_matrix[counter] = layer
        else:
            layer = np.append(
                layer, np.zeros(max_number_of_nodes - len(layer)) + np.nan
            )
            square_matrix[counter] = layer
        counter += 1
    return square_matrix


def plot_NAP(data: np.array, output_path: str) -> None:
    nodes_numbers = np.array([0, 50, 100, 200])
    fig, ax = plt.subplots()
    NAP = ax.imshow(
        data.T,
        cmap="RdBu_r",
        interpolation="nearest",
        aspect="auto",
        origin="lower",
        norm=matplotlib.colors.CenteredNorm(),
    )
    colorbar = plt.colorbar(NAP)
    colorbar.set_label("Activation")
    ax.set_title("Neural Activation Pattern")
    ax.set_xlabel("Layers")
    ax.set_ylabel("Number of nodes")
    xtick_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(xtick_loc))
    ax.set_xticklabels(["", "en1", "en2", "en3", "de1", "de2", "de3", ""])
    ax.set_yticks(nodes_numbers)
    ax.figure.savefig(os.path.join(output_path, "diagnostics.pdf"))

def plot_corr(input_path: str, output_path: str, config):
    before_path = config.input_path
    after_path = os.path.join(input_path, "decompressed_output", "decompressed.npz")
    corr_path = os.path.join(input_path, "decompressed_output", "correlations")

    before = np.transpose(np.load(before_path)["data"])
    after = np.transpose(np.load(after_path)["data"])
    names = np.load(config.input_path)["names"]

    index_to_cut = plotting.get_index_to_cut(3, 1e-6, before)
    before = np.delete(before, index_to_cut, axis=1)
    after = np.delete(after, index_to_cut, axis=1)

    corr_before = np.corrcoef(before)
    corr_after = np.corrcoef(after)
    corr_diff = np.abs(corr_after - corr_before)
    np.save(corr_path, corr_diff)

    column_names = [i.split(".")[-1] for i in names]

    with PdfPages(os.path.join(output_path, "correlations.pdf")) as pdf:
        fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 8), gridspec_kw={'height_ratios': [1]})
        im = ax1.imshow(
            corr_before,
            vmin=-1, vmax=1,
            cmap="RdBu_r"
        )
        im = ax2.imshow(
            corr_after,
            vmin=-1, vmax=1,
            cmap="RdBu_r"
        )

        # Formatting for original values corr matrix
        ax1.set_xticks(range(len(corr_before[0])))
        ax1.set_yticks(range(len(corr_before[0])))
        ax1.set_xticklabels(column_names)
        ax1.set_yticklabels(column_names)
        ax1.set_title("Original variables\n", fontsize=14)
        plt.setp(ax1.get_xticklabels(), rotation=270, ha='left', rotation_mode='anchor')

        # Formatting for decompressed vals corr matrix
        ax2.set_xticks(range(len(corr_after[0])))
        ax2.set_yticks([])
        ax2.set_xticklabels(column_names)
        ax2.set_yticklabels([])
        ax2.set_title("Decompressed variables\n", fontsize=14)
        plt.setp(ax2.get_xticklabels(), rotation=270, ha='left', rotation_mode='anchor')

        fig1.tight_layout()
        fig1.subplots_adjust(right=0.825)
        cax1 = fig1.add_axes([0.85, 0.335, 0.035, 0.47])
        cb1 = fig1.colorbar(im, cax=cax1)
        cb1.set_label("Correlation", fontsize=10)
        pdf.savefig(fig1)

        fig2, ax = plt.subplots(figsize=(12,11))
        im_diff = ax.imshow(
            corr_diff
        )
        ax.set_xticks(range(len(corr_diff[0])))
        ax.set_yticks(range(len(corr_diff[0])))
        ax.set_xticklabels(column_names, fontsize=14)
        ax.set_yticklabels(column_names, fontsize=14)
        ax.set_title(
            "Absolute value of the difference between\n correlations of original and decompressed variables\n",
            fontsize=20,
            pad=1.0)
        plt.setp(ax.get_xticklabels(), rotation=270, ha='left', rotation_mode='anchor')

        fig2.tight_layout()
        fig2.subplots_adjust(right=0.85)
        cax2 = fig2.add_axes([0.85, 0.2775, 0.035, 0.625])
        cb = fig2.colorbar(im_diff, cax=cax2)
        cb.ax.tick_params(labelsize=14)
        cb.set_label("Correlation", fontsize=14)
        pdf.savefig(fig2)





def diagnose(input_path: str, output_path: str, config) -> None:
    activations_path = os.path.join(input_path, "training", "activations.npy")
    input = np.load(activations_path)
    plot_NAP(input, output_path)
    plot_corr(input_path, output_path, config)
    print(
        "Diagnostics saved as diagnostics.pdf in the plotting folder of your project."
    )
