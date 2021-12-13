
Image Colorization using Self-attention Conditional GAN with Content Loss

Summary:

This challenge aims to leverage learning-based techniques to train machine learning or deep learning models for image colorization. Inspired by previous work and a tutorial [1, 2], I also a deep learning model to learn a color distribution for a given set of color images and use the learned model to color grayscale images. Specifically, I implement a conditional generative adversarial network with a self-attention layer that generates features of the salient areas in a generator to produce synthetic color images given grayscale images. I also leverage a perceptual loss to enforce that the generated color images should be close to actual color images. In measuring the difference between the fake and ground-true color images, I use PSNR and SSIM to conduct the measurement because these two metrics are commonly used in image quality measurements.

Inspired by the work [1], I adopt a conditional generative adversarial network (conditional GAN) as a base model for this image colorization problem. Since our goal is to train a generator that can synthesize color using a set of color images without collecting additional grayscale images, I follow the existing strategy [1, 2] to convert RGB color images to Lab color images. The generator then has the capability of learning the green-read opponent colors (a) and blue-yellow opponent colors (b) using the lightness value (L). The advantage is that I do not need to create the additional images and simplify the framework for model training. In addition, a self-attention layer is utilized in my GAN model because this layer capably learns complementary and consistent features that represent objects or scenarios in images [3]. Finally, in terms of loss functions in this challenge, I combine with three losses: an adversarial loss, an L-1 loss that takes groud-true images and synthetic images as inputs, and a perceptual loss that takes groud-true images and synthetic images as inputs.

The future work of this research topic can have two directions. First, I can integrate color priors manually (user selection) or automatically (prior generation via other deep neural networks [4]) into the current GAN model. The generator can thus control how to learn a variety of colors and produce high-quality color images for users. Second, for the experiment metrics, I can use more precise metrics to evaluate the quality of the synthetic color images using color-based reference and non-reference metrics for image assessment measurement tasks.

Reference:
[1] https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8
[2] https://richzhang.github.io/colorization/
[3] Self-Attention Generative Adversarial Networks (https://arxiv.org/pdf/1805.08318.pdf)
[4] A Style-Based Generator Architecture for Generative Adversarial Networks (https://arxiv.org/abs/1812.04948)

Prerequisites:

Python3
PyTorch (These implementations have been tested on PyTorch 1.7.0)
Torchvision
PIL
scikit-image (skimage)
Numpy
pathlib
glob


Get Started:
Preparation:

Install all prerequisites before running any scripts.

Put your data with color images in any places you prefer. I recommand you can put the data in the image-coloring directory.

Note: (1) you do not need to split data into training and val data since the script uses a default ratio to generate two datasets automatically during training. You can change the ratio value via the --splitPercent argument. (2) The scripts only work on training, validation, and test data with jpg images. Please convert all images to the jpg format before you start running model training and model inference. (3) An example dataset: https://drive.google.com/file/d/15jprd8VTdtIQeEtQj6wbRx6seM8j0Rx5/view

Model Training:

To do model training of a self-attention conditional GAN on training data in a single GPU machine, please run:

python train.py --dataroot="path to a directory with training and val color images"

E.g., python train.py --dataroot=./landscape-images/

Note: this script uses default hyper-parameters for model training. To change the hyper-parameters, please set --lambdaL1, --lambdaPerceptual, and --epoch accordingly.


Model Inference:

Once model training is done, you can run the following command to generate synthetic color images:

(1) Given a val dataset with color images, please use the --dataroot argument for the input directory path

python test.py --dataroot="path to a directory with training and val color images"

E.g., python test.py --dataroot=./landscape-images/

(2) Given a test dataset with grayscale images, please use the --testdataroot argument for the input directory path

python test.py --testdataroot="path to a directory with test grayscale images"

E.g., python test.py --testdataroot=./test-data/



