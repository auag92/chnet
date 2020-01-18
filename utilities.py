import numpy as np
import matplotlib.pyplot as plt


def draw(X, title='', sample_id = 0):
    if X.ndim == 3:
        try:
            im = plt.imshow(X[sample_id][:][:], extent=[0, 1, 0, 1])
            plt.colorbar()
            plt.title(title)
            plt.show(im)
        except IndexError:
            print("Requested Sample # %d exceeds data size" % (sample_id+1))
    else:
        im = plt.imshow(X[:][:], extent=[0, 1, 0, 1])
        plt.colorbar()
        plt.title(title)
        plt.show(im)


def draw_by_side(X, Y, title='', title_left='', title_right='', sample_id = 0):
    if X.ndim == 3:
        try:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            fig.suptitle(title, fontsize=20)

            ax1.set_title(title_left)
            im1 = ax1.imshow(X[sample_id],interpolation='nearest')

            ax2.set_title(title_right)
            im2 = ax2.imshow(Y[sample_id],interpolation='nearest')

            plt.tight_layout()
            plt.show()
        except IndexError:
            print("Requested Sample # %d exceeds data size" % (sample_id+1))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        fig.suptitle(title, fontsize=20)

        ax1.set_title(title_left)
        im1 = ax1.imshow(X,interpolation='nearest')

        ax2.set_title(title_right)
        im2 = ax2.imshow(Y,interpolation='nearest')

        plt.tight_layout()
        plt.show()
