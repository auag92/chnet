import numpy as np
import matplotlib.pyplot as plt
from toolz.curried import pipe, curry, compose
from mpl_toolkits.axes_grid1 import make_axes_locatable

def draw_im(im, title=None):
    im = np.squeeze(im)
    plt.imshow(im)
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()
    
@curry
def return_slice(x_data, cutoff):
    if cutoff is not None:
        return pipe(x_data,
                    lambda x_data: np.asarray(x_data.shape).astype(int) // 2,
                    lambda new_shape: [slice(new_shape[idim]-cutoff,
                                             new_shape[idim]+cutoff+1)
                                       for idim in range(x_data.ndim)],
                    lambda slices: x_data[slices])
    else:
        return x_data
    

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

def colorbar(mappable):
    """
    https://joseph-long.com/writing/colorbars/
    """
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar
        
        
def draw_by_side(args, title='', sub_titles =["", "", ""], scale=6):
    fig, axs = plt.subplots(nrows=1, ncols=len(args), figsize=(scale*1.6180,scale))
    fig.suptitle(title, fontsize=20)
    
    for ix, arg in enumerate(args):
        axs[ix].set_title(sub_titles[ix])
        im1 = axs[ix].imshow(arg, interpolation='nearest', cmap='seismic', vmin=-1., vmax=1.0)
        colorbar(im1)
        
    for ix, ax in enumerate(axs[1:]):
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
        
        
        
# def draw_by_side(X, Y, title='', title_left='', title_right='', sample_id = 0):
#     if X.ndim == 3:
#         try:
#             fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
#             fig.suptitle(title, fontsize=20)

#             ax1.set_title(title_left)
#             im1 = ax1.imshow(X[sample_id],interpolation='nearest')

#             ax2.set_title(title_right)
#             im2 = ax2.imshow(Y[sample_id],interpolation='nearest')

#             plt.tight_layout()
#             plt.show()
#         except IndexError:
#             print("Requested Sample # %d exceeds data size" % (sample_id+1))
#     else:
#         fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
#         fig.suptitle(title, fontsize=20)

#         ax1.set_title(title_left)
#         im1 = ax1.imshow(X,interpolation='nearest')

#         ax2.set_title(title_right)
#         im2 = ax2.imshow(Y,interpolation='nearest')

#         plt.tight_layout()
#         plt.show()
