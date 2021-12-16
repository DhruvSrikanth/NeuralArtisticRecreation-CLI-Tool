# Project - "Neural Artistic Recreation"

1. Project Idea - I would like to create an API to enable people to experience the art of their choosing on a more personal level.

2. Project To Do's - 
	1. Convert an input image to an stylized image by the artist and in the theme of their choosing. The style (artist style and color style) will be used in recreating the input image as a stylized output. This can be done through neural style transfer.

	2. Apply certain filters (like invert, black-and-white) to the stylized image to further customize the art to the users choosing. 

3. Project Execution Plan - 
	1. By the end of week 1, I will have explored methods for neural style transfer and methods to apply certain filters on images.
	2. By the end of week 2, I will have created a pipeline that takes an input image and outputs an image with a filter (by choice) applied to it.
	3. By the end of week 3, I will have created a pipeline that performs neural style transfer (users will use the style image of their choice and the input image of their choice in the pipeline).
	4. By the end of week 4, I will have integrated both of the above pipelines.
	5. By the end of week 5, I will have created a module/package for the entire pipeline. 
	6. Finally, I will include some "popular" style images that users can use if they do not have a style image in mind.

4. Project Demo - 
[Neural Artistic Recreation](https://youtu.be/AJCFO6ot4B8)

---

# User CLI Guide

<br>

This tool aims to create an API to enable people to experience the art of their choosing on a more personal level using filters from the **Python Imaging Library** and **Neural Style Transfer**.

<br>

---

<H3>Video Demo:</H3>


[Neural Artistic Recreation - Video Demo](https://youtu.be/AJCFO6ot4B8)


<H3>Sample:</H3>

There are samples for an input and corresponding output of the model with different styles present in the **samples** folder.

<H3>References</H3>

<br>

Given below are some of references that can be used for better understanding some of the underlying concepts and code used in this tool:  

1. Python Imaging Library: <https://pillow.readthedocs.io/en/stable/>
2. Neural Style Transfer: <https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html> 
3. Style Transfer with Tensorflow: <https://www.tensorflow.org/tutorials/generative/style_transfer>
4. Transfer Learning: <https://www.tensorflow.org/tutorials/images/transfer_learning>

<br>

---

<H3>Requirements</H3>  

All requirements can be found in the **requirements.txt** file. I would recommend running commenting out all tensorflow and keras related lines as the version required for these packages differs from system to system. For tensorflow and keras, please refer to tensorflow's guide on installing tensorflow, cuda and cudnn for the correct versions for your system. Furthermore, before installing all the requirements, I would recommend creating a virutal environment. I use [venv](https://docs.python.org/3/library/venv.html#module-venv) for this. After pulling this repository into a particular directory of your choice, you may activate your virtual environment and run the following command to install all of the required packages - 

```
pip install -r /path/to/requirements.txt
```

I recommend using a system with an powerful NVIDIA GPU if you want to train the model for a higher number of epochs if you would like to further refine the output.

<br>

---


<H3>Using the Tool</H3>  

Run the below command to use the tool - 

```
python main.py --inp_img_path=/path/to/input_image --style_choice=choice_of_style --filter_choice=choice_of_filter --save_path=/path/to/stylized_image"
```

<H4>Filter Choices:</H4>  

1. No Filter
2. Blur
3. Contour
4. Detail
5. Edge Enhance
6. Strong Edge Enhance
7. Emboss
8. Find Edges
9. Smooth
10. Strong Smooth
11. Sharpen

More on these filters can be found [here](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html).

<H4>Style Choices:</H4>  

1. No Style
2. [Piccaso](https://www.wikiart.org/en/pablo-picasso/self-portrait-1907)
3. [Van Gogh](https://en.wikipedia.org/wiki/The_Starry_Night#/media/File:Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg)
4. [Michelangelo](https://en.wikipedia.org/wiki/The_Creation_of_Adam#/media/File:Michelangelo_-_Creation_of_Adam_(cropped).jpg)
5. [Leonardo Da Vinci](https://en.wikipedia.org/wiki/Mona_Lisa#/media/File:Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg)
6. [Edward Munch](https://upload.wikimedia.org/wikipedia/commons/8/86/Edvard_Munch_-_The_Scream_-_Google_Art_Project.jpg)
7. [Sandro Botticelli](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg/2560px-Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg)

These styles can also be viewed in the **styles** directory.

<H4>Customization:</H4>  

If you would like to add more styles, this can be done in the following way:

1. Download the style image of your choice and move this to the **styles** directory.
2. Open the **styles.py** file inside **src**.
3. Add a path variable containing the path to the new style image to the file.
4. Add this variable to the end of the **STYLES** list.

If you would like to refine your output, this can be done in the following way:

1. Open the **parameters.py** file inside **src**.
2. Change the **EPOCHS** and **STEPS_PER_EPOCH** variable values. The higher these values are, the more time the tool will take to run as the model will train for a longer period of time, however, this will result in a more refined output.

If you would like to test out different weights for the different losses or tune the hyperparameters further, this can be done in the following way:

1. Open the **parameters.py** file inside **src**.
2. Change the parameters and/or hyperparameters to the values you would like to test with the model.



<br>
