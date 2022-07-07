"""
Class for creating various plots

"""
from os.path import join
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager
from matplotlib import rc
from sklearn.manifold import TSNE
import umap
from collections import OrderedDict
from ..utils.calc_utils import calc_corr
import itertools

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
plt.switch_backend("Agg")


class Plotting:
    def __init__(self, *path):
        if path:
            self.output_path = path

    def plot_losses(self, logger, title=""):
        plt.figure()
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        plt.subplot(1, 2, 1)
        plt.title("Loss values")
        for k, v in logger.logs.items():
            plt.plot(v, label=str(k))
        plt.xlabel(r"\textbf{epochs}", fontsize=10)
        plt.ylabel(r"\textbf{loss}", fontsize=10)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.title("Loss relative values")
        for k, v in logger.logs.items():
            max_loss = 1e-8 + np.max(np.abs(v))
            print(max_loss)
            plt.plot(v / max_loss, label=str(k))
        plt.legend()
        plt.xlabel(r"\textbf{epochs}", fontsize=10)
        plt.ylabel(r"\textbf{loss}", fontsize=10)
        plt.savefig(join(self.output_path, "Losses{0}.png".format(title)))
        plt.close()

    def plot_tsne(self, data, target, title, title_short):
        for i, data_ in enumerate(data):
            tsne_model = TSNE(n_components=2)
            projections = tsne_model.fit_transform(data_)
            color_code = ["b", "r", "c", "m", "y", "g", "k", "orange", "pink", "gray"]
            plt.figure()
            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")
            for j in range(len(np.unique(target))):
                plt.plot(
                    projections[np.where(target == j), 0],
                    projections[np.where(target == j), 1],
                    color=color_code[j],
                    label=str(j),
                    linestyle="None",
                    marker=".",
                )
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.title("t-SNE {0}".format(title[i]))
            plt.savefig(
                join(self.output_path, "tSNE_view_{0}.png".format(title_short[i]))
            )
            plt.close()

    def plot_UMAP(self, data, target, title, title_short):
        for i, data_ in enumerate(data):
            reducer = umap.UMAP()
            projections = reducer.fit_transform(data_)
            color_code = [
                "b",
                "r",
                "c",
                "m",
                "y",
                "g",
                "k",
                "orange",
                "pink",
                "gray",
                "b",
                "r",
                "c",
                "m",
                "y",
                "g",
                "k",
                "orange",
                "pink",
                "gray",
                "b",
                "r",
                "c",
                "m",
                "y",
                "g",
                "k",
                "orange",
                "pink",
                "gray",
            ]
            plt.figure()
            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")
            for j in range(len(np.unique(target))):
                plt.plot(
                    projections[np.where(target == j), 0],
                    projections[np.where(target == j), 1],
                    color=color_code[j],
                    label=str(j),
                    linestyle="None",
                    marker=".",
                )
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.title("UMAP {0}".format(title[i]))
            plt.savefig(join(self.output_path, "UMAP_{0}.png".format(title_short[i])))
            plt.close()

    def plot_dropout(self, title=""):
        if self.sparse:
            do = np.sort(self.dropout().cpu().detach().numpy().reshape(-1))
            print(do)
            plt.figure()
            plt.bar(range(len(do)), do)
            plt.title("Dropout probability of {0} latent dimensions".format(self.z_dim))
            plt.savefig(join(self.output_path, "dropout{0}.png".format(title)))

    def print_reconstruction(self, *data, recon_type=None, save=True):
        print("~~~~~~~printing reconstruction results~~~~~~~")
        x_recon = self.predict_reconstruction(*data)
        to_print = []
        if self.joint_representation:
            recon_loss = 0
            for i in range(self.n_views):
                recon_loss_temp = np.mean((x_recon[i] - data[i]) ** 2)
                recon_loss += recon_loss_temp
                to_print.append(
                    "Same view reconstruction on {0} data for view {1}: {2}".format(
                        recon_type, i, recon_loss_temp
                    )
                )
            recon_loss = recon_loss / self.n_views
            to_print.append(
                "Average same view reconstruction on {0} data: {1}".format(
                    recon_type, recon_loss
                )
            )
            recon_loss = 0
            for i in range(self.n_views):
                data_input = list(data)
                data_input[i] = np.zeros(np.shape(data[i]))
                x_recon = self.predict_reconstruction(*data_input)
                recon_loss_temp = np.mean((x_recon[i] - data[i]) ** 2)
                recon_loss += recon_loss_temp
                to_print.append(
                    "Cross view reconstruction on {0} data with for view {1} with view {1} missing: {2}".format(
                        recon_type, i, recon_loss_temp
                    )
                )
            recon_loss = recon_loss / self.n_views
            to_print.append(
                "Average cross view reconstruction on {0} data: {1}".format(
                    recon_type, recon_loss
                )
            )
        else:
            same_view_recon = 0
            cross_view_recon = 0
            counter = 0
            for i in range(self.n_views):
                for j in range(self.n_views):
                    if i == j:
                        same_view_temp = np.mean((x_recon[i][j] - data[i]) ** 2)
                        same_view_recon += same_view_temp
                        counter += 1
                        to_print.append(
                            "Same view reconstruction on {0} data for view {1}: {2}".format(
                                recon_type, i, same_view_temp
                            )
                        )
                    else:
                        cross_view_temp = np.mean((x_recon[i][j] - data[i]) ** 2)
                        cross_view_recon += cross_view_temp
                        counter += 1
                        to_print.append(
                            "Cross view reconstruction on {0} data for view {1}: {2}".format(
                                recon_type, i, cross_view_temp
                            )
                        )
            print("counter: ", counter)
            same_view_recon, cross_view_recon = (
                same_view_recon / self.n_views,
                cross_view_recon / self.n_views,
            )
            to_print.append(
                "Average same view reconstruction on {0} data: {1}".format(
                    recon_type, same_view_recon
                )
            )
            to_print.append(
                "Average cross view reconstruction on {0} data: {1}".format(
                    recon_type, cross_view_recon
                )
            )

    def plot_corr(self, *latents, corr_type="pearson"):
        if self.joint_representation:
            print("Cannot create correlation plot from joint latent representation")
            return
        # TO DO: need to check if this works
        combs = list(itertools.combinations(range(self.n_views), 2))
        for comb in combs:
            # print("corr combination: ", comb)
            i, j = comb[0], comb[1]
            data_1, data_2 = latents[i], latents[j]

            data_1_T = np.transpose(data_1)
            data_2_T = np.transpose(data_2)
            corr = calc_corr(data_1_T, data_2_T, corr_type="pearson")
            # rows = x
            rows = ["view 1 vec %d" % x for x in range(self.z_dim)]
            # columns = y
            columns = ["view 2 vec %d" % x for x in range(self.z_dim)]
            fig, ax = plt.subplots()
            data = np.round(corr[0 : self.z_dim, self.z_dim :], 4)
            data = data / np.mean(data)
            data = np.round(data, 4)
            im = ax.imshow(abs(data), cmap="Reds")
            ax.set_xticks(np.arange(len(rows)))
            ax.set_yticks(np.arange(len(columns)))
            ax.set_xticklabels(rows)
            ax.set_yticklabels(columns)
            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    text = ax.text(j, i, data[i, j], ha="center", va="center")
            ax.set_title("Cross correlation")
            fig.tight_layout()
            plt.savefig(
                join(
                    self.output_path,
                    "Cross_corr_plot_views{0}_{1}_training_relative.png".format(i, j),
                )
            )

    def proj_latent_corr(self, latent, projection, title):
        print(
            "~~~~correlations between PLS projections and VAE latent space results~~~~"
        )
        print("Corr with {0} PLS projection 1".format(title))
        for i in range(self.z_dim):
            corr = calc_corr(latent[:, i], projection[:, 0], corr_type="pearson")
            print("corr for vec {0}: ".format(i), corr[0, 1])
        print("Corr with {0} PLS projection 2".format(title))
        for i in range(self.z_dim):
            corr = calc_corr(latent[:, i], projection[:, 1], corr_type="pearson")
            print("corr for vec {0}: ".format(i), corr[0, 1])
