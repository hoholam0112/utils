import matplotlib.pyplot as plt

def subplot_image(imgs, shape, fig_name, show=True, save=False, save_dir=None, centered=True):
    """
    Args:
        imgs: np.array with shape of (nb_imgs, height, width, channel) or (nb_imgs, height, width)
        shape: (nb_row, nb_col)
        save (boolean): save the figure or not
        show (boolean): show the figure or not
    """
    assert len(shape) == 2, 'Length of \'shape\' must be 2'
    assert len(imgs.shape) == 3 or len(imgs.shape) == 4
    assert imgs.shape[0] == shape[0]*shape[1], 'Number of subplots are not matched to number of images'
    assert save == bool(save_dir), 'Maybe \'save_dir\' takes no argument when \'save\' is given True'

    nb_row, nb_col = shape
    plt.figure()
    plt.title(str(fig_name))
    for i in range(imgs.shape[0]):
        plt.subplot(nb_row, nb_col, i+1)
        if centered:
            _imgs = (imgs+1)/2
        else:
            _imgs = imgs

        if len(_imgs.shape) == 3:
            plt.imshow(_imgs[i], cmap='gray')
        else:
            plt.imshow(_imgs[i])

        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
            labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)

    if save:
        plt.savefig('{}.pdf'.format(os.path.join(save_dir, fig_name)), format='pdf',
            transparent=True, frameon=False)

    if show:
        plt.show()
    else:
        plt.close()


