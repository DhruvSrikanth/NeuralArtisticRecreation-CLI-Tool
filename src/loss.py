import tensorflow as tf

import parameters

def total_loss(model_obj, stylized_image, content_image, style_image):
    """
    Compute total loss inclusive of style loss, content loss and variation loss.

    :param model_obj: instance of the model being trained
    :param stylized_image: desired stylized target image
    :param content_image: content image being used
    :param style_image: style image being used
    :returns loss: total loss between stylized image and content and style images

    """
    outputs = model_obj(stylized_image)

    # Style loss
    style_targets = model_obj(style_image)['style']
    style_outputs = outputs['style']
    style_loss = tf.add_n([tf.reduce_mean((style_targets[name]-style_outputs[name])**2) for name in style_outputs.keys()])
    style_loss *= parameters.STYLE_WEIGHT / model_obj.num_style_layers

    # Content loss
    content_targets = model_obj(content_image)['content']
    content_outputs = outputs['content']
    content_loss = tf.add_n([tf.reduce_mean((content_targets[name]-content_outputs[name])**2) for name in content_outputs.keys()])
    content_loss *= parameters.CONTENT_WEIGHT / model_obj.num_content_layers

    # Variation loss
    variation_loss = parameters.VARIATION_WEIGHT*tf.image.total_variation(stylized_image)

    loss = style_loss + content_loss + variation_loss

    return loss