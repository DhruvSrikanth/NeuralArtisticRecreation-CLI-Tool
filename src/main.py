import time
import argparse

from PIL import Image

import tensorflow as tf

import utils
import filters
import styles
import model
import train


# Argument Parser
parser = argparse.ArgumentParser(description='Neural Artistic Recreation')

parser.add_argument('--inp_img_path', type=str, required=True, help='Specify the input image path.')
parser.add_argument('--style_choice', type=int, required=True, help='Specify the style choice out of the following artists: \n 1. No Style \n 2. Piccaso \n 3. Van Gogh \n 4. Michelangelo \n 5. Leonardo Da Vinci \n 6. Edward Munch \n 7. Sandro Botticelli')
parser.add_argument('--filter_choice', type=int, required=True, help='Specify the style choice out of the following artists:  \n 1. No Filter \n 2. Blur \n 3. Contour \n 4. Detail \n 5. Edge Enhance \n 6. Strong Edge Enhance \n 7. Emboss \n 8. Find Edges \n 9. Smooth \n 10. Strong Smooth \n 11. Sharpen')
parser.add_argument('--save_path', type=str, required=True, help='Specify the output image path.')

args = parser.parse_args()

inp_img_path = args.inp_img_path
style_choice = args.style_choice
if style_choice > len(styles.STYLES) + 1 or style_choice < 1:
    raise NotImplementedError("Style selected is not present...Please look at the available styles and pick the style using the appropriate number.")
filter_choice = args.filter_choice
if filter_choice > len(filters.FILTERS) + 1 or filter_choice < 1:
    raise NotImplementedError("Filter selected is not present...Please look at the available filters and pick the filter using the appropriate number.")
save_path = args.save_path


content_path = utils.convert_to_png(inp_img_path)
if style_choice != 1:
    style_path = utils.convert_to_png(styles.STYLES[style_choice - 2])
    # Style image
    style_image = utils.load_img(style_path)
    # utils.disp_img(style_image, 'Style Image')

# Content image
content_image = utils.load_img(content_path)
# utils.disp_img(content_image, 'Content Image')

if style_choice != 1:
    # Training
    stylized_image = tf.Variable(content_image)
    start = time.time()
    computed_loss = train.train_step(model.extractor, stylized_image, content_image, style_image)
    end = time.time()
    print("Total time: {:.1f}".format(end-start))

    # Save stylized image
    utils.tensor_to_image(stylized_image).save(save_path)
else:
    stylized_image = tf.Variable(content_image)
    utils.tensor_to_image(stylized_image).save(save_path)

if filter_choice != 1:
    # Apply Filter
    stylized_image = Image.open(save_path)
    filtered_image = stylized_image.filter(filters.FILTERS[filter_choice - 2])

    # Save stylized image
    filtered_image.save(save_path)
else:
    stylized_image = Image.open(save_path)
    stylized_image.save(save_path)


