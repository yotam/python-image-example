"""
Demo of Python/NumPy "best practices"

* basic path manipulation in Python with the os module and glob
* string formatting
* reading and writing images
* Using NumPy functions on an array
* array creation with NumPy broadcasting
* converting between data types

"""

import numpy as np
from skimage.io import imread, imsave
from skimage import img_as_float, img_as_ubyte

# alternatively:
# from scipy.misc import imread, imsave
# but the skimage version will also read from a URL

from glob import glob
import os

def vignette(image):
    """
    Returns a vignetted version of the image

    Parameters
    ----------
    image: ndarray
        RGB image

    Returns
    -------
    vignetted_image: ndarray
    
    Notes
    -----
    http://en.wikipedia.org/wiki/Vignetting
    """
    # every numpy array has a shape tuple
    width, height = image.shape[:2]

    # compute the vignette
    # first calculate the squared distance from the center of the image

    # this uses broadcasting (the [..., np.newaxis] syntax) to construct a 
    # 2d array from two 1d arrays
    #   * http://scipy-lectures.github.io/intro/numpy/operations.html#broadcasting

    # broadcasting is used to achieve the same things as meshgrid, repmat and bsxfun in Matlab
    # but in theory more efficiently/concisely

    xs = np.arange(width)
    ys = np.arange(height)
    distance_squared = (xs - width/2.0)[..., np.newaxis] ** 2 + (ys - height/2.0) ** 2

    # like Matlab, there's a vectorized version of most mathematical functions
    sigma_squared = (width/2.0) ** 2 + (height/2.0) ** 2
    falloff = np.exp(-distance_squared/sigma_squared)

    # again use broadcasting to apply this to each channel of the image
    # the image has shape (width, height, channels)
    # but the falloff array has shape (width, height)
    # if we tried to do image * falloff, we would get an error
    # with broadcasting we change the falloff array shape into (width, height, 1)
    # and the operation works
    result = image * falloff[..., np.newaxis]

    # result will be an array of floats, however it's good practice to explicitly
    # cast to uint8 for saving
    return img_as_ubyte(result)


def main():
    # get a list of image filenames in a directory
    base_directory = "/path/to/image/directory"

    # create an output_directory
    output_directory = os.path.join(base_directory, "output")
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # string formatting with leading zeros - we'll use this later
    output_filename_template = os.path.join(output_directory, "output {:03d}.png")

    # this will work if base_directory does not end with '/'
    filenames = glob(base_directory + "/*.png")

    # but it's better to safely join paths
    filenames = glob(os.path.join(base_directory, "*.png"))

    # will often want to process these in order
    filenames = sorted(glob(os.path.join(base_directory, "*.png")))

    # process each image
    for i, filename in enumerate(filenames):
        image = img_as_float(imread(filename))
        result = vignette(image)
        imsave(output_filename_template.format(i), result)

if __name__ == "__main__":
    main()
