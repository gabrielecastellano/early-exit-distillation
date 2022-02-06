import os

import matplotlib.pyplot as plt
import numpy as np


def intermediate_output_to_fig(z, img_ctr, label, predicted, confidence):
    n_columns = 4
    n_rows = 3
    f, ax = plt.subplots(n_rows, n_columns)
    f.tight_layout()
    for i, channel_o in enumerate(z):
        c_img = channel_o * 1
        r, c = divmod(i, n_columns)
        ax[r][c].set_title('Channel ' + str(i))
        # ax[r][c].xticks([])
        # ax[r][c].yticks([])
        low = np.quantile(c_img, 0.01)
        high = np.quantile(c_img, 0.99)
        im = ax[r][c].imshow(c_img, vmin=low, vmax=high, interpolation=None)
        f.colorbar(im, ax=ax[r][c])
    if not os.path.exists(f"bottleneck_output/7b-3ch/{label}"):
        os.mkdir(f"bottleneck_output/7b-3ch/{label}")
    f.savefig(f"bottleneck_output/7b-3ch/{label}/{predicted}{img_ctr}_conf={confidence}_3.png")
    plt.close()
