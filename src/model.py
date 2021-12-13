import tensorflow as tf

import utils
import parameters


# Check to see what layers are needed for our model
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# Check layers present
# for layer in vgg.layers:
#   print(layer.name)


# Required content and style layers
content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']


# Base model
def vgg_layers(layer_names):
    """
    Creates a vgg model that returns a list of intermediate output values.

    :param layer_names: names of layers that are to be used in the model.
    :returns: return the VGG-based portion of the model to be used for transfer learning.

    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)

    return model

# Custom Model
class StyleTransferModel(tf.keras.models.Model):
    """
    Model that is used for Style Transfer.
    
    """

    def __init__(self, style_layers, content_layers):
        super(StyleTransferModel, self).__init__()
        self.vgg =  vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """

        :param inputs: input to the model (preprocessed images)

        """
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)

        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        style_outputs = [utils.gram_matrix(style_output)for style_output in style_outputs]

        content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content':content_dict, 'style':style_dict}

# Build model
extractor = StyleTransferModel(style_layers, content_layers)


# Optimizer used
opt = tf.optimizers.Adam(learning_rate=parameters.LEARNING_RATE, beta_1=parameters.BETA_1, epsilon=parameters.EPSILON)