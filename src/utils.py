import PIL.Image
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

def convert_to_png(path):
    """
    Convert image to png format.

    :param path: path of image to be converted to png format
    :returns: path of converted image
    :raises: NotImplementedError when a format other than jpg, jpeg, and png are used for the input content and style images

    """
    extension = path.split('.')[-1].lower()
    if extension == 'jpg' or extension == 'jpeg' or extension == 'png':
        if extension != 'png':
            im = PIL.Image.open(path)
            new_path = path[:-len(extension)] + 'png'
            im.save(new_path)
            return new_path
        return path
    else:
        raise NotImplementedError


def load_img(path_to_img):
    """
    Preprocess image.

    :param path_to_img: path of input image
    :returns: preprocessed image

    """
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img


def disp_img(image, title=None):
    """
    Display image.

    :param image: input image to be displayed
    :param title:  title to be displayed for the input image (Default value = None)

    """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    plt.show()
    if title:
        plt.title(title)



def tensor_to_image(tensor):
    """
    Convert tensor to image.

    :param tensor: tensor object to be converted
    :returns: image object

    """
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
          
    return PIL.Image.fromarray(tensor)

def gram_matrix(input_tensor):
    """
    Computes gram matrix.

    :param input_tensor: input tensor from which the gram matrix will be calculated
    :returns: gram matrix

    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)

    return result/(num_locations)