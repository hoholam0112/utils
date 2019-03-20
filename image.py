import os, cv2
import numpy as np
from PIL import Image

def resize(image, size):
    """ Load and resize image
    Args:
        image: uint8 numpy array with shape (height, width, channel)
        size: int or tuple of int, (height, width)
    Returns:
        resized image: numpy array (height, width, channel)
    """
    if isinstance(size, int):
        size = (size, size)
    else:
        size = (size[1], size[0])
    return cv2.resize(image, size)

def save(fname, image):
    """ Save image as a designated format
        Args:
            fname: str, path to save image
            image: numpy array (height, weight, channel)
        Returns:
            None
    """
    iobj = Image.fromarray(image)
    iobj.save(fname)

def imread(fname):
    """ Read image from disk
        Args:
            fname: str, path to an image file
        Returns:
            image: unit8 numpy array
    """
    iobj = Image.open(fname)
    image = np.asarray(iobj, dtype=np.uint8)
    return image

