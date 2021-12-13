import tensorflow as tf
from tqdm import tqdm

import parameters
import loss
import model

def train_step(model_obj, stylized_image, content_image, style_image):
    """Function to train the model.

    :param model_obj: instance of the model being trained
    :param stylized_image: desired stylized target image
    :param content_image: content image being used
    :param style_image: style image being used
    :returns loss: total loss between stylized image and content and style images

    """
    epochs = parameters.EPOCHS
    steps_per_epoch = parameters.STEPS_PER_EPOCH
    for epoch in tqdm(range(epochs), desc='Model Training Progress', leave = False):
    	for step in range(steps_per_epoch):
    		with tf.GradientTape() as tape:
    			computed_loss = loss.total_loss(model_obj, stylized_image, content_image, style_image)
    		grad = tape.gradient(computed_loss, stylized_image)
    		model.opt.apply_gradients([(grad, stylized_image)])
    		stylized_image.assign(tf.clip_by_value(stylized_image, clip_value_min=0.0, clip_value_max=1.0))

    return computed_loss