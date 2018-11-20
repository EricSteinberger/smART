# smART - the robotic artist

smART is an industrial robot arm painting in an artistic style. Watch a
[timelapse](https://www.youtube.com/watch?v=AB6-94rLlVI) of the painting process here! :)

![smART painting process](http://www.steinberger-ai.com/wp-content/uploads/2018/07/smART_titel_wide-e1532988954888.jpg)

## Details
This repository consists of these components:
* **Learning brush strokes** - A library of strokes (movement+image pairs) is created interactively
* **Neural Style Transfer [[Gatys et al.]](https://arxiv.org/pdf/1508.06576.pdf)** - The robot's own style is applied to every image it shall paint.
* **Iterative Stroke Sampling** - ISS combines strokes in the library to approximate the stylized image in a simulation.
* **ABB RAPID code interface** - The sequence of commands returned by ISS is translated to RAPID code
(ABB's robot programming language).

The primary purpose of this repository is to provide insight into how smART works and to provide implementations of 
Neural Style Transfer (NST) and Iterative Stroke Sampling (ISS) to play with.

## Getting Started
This section will show you how you can play around with NST and ISS. Note that you will likely not have much use for the
RAPID command sequences ISS produces as these are quite specific to our setup.

### Prerequisites
This codebase is tested on Ubuntu 16.04 and Windows 10 with Python 3.7. A GPU will give you a speed-up on NST and ISS alike, but the algorithms will also work on your CPU.

### Installing
I recommend using Anaconda as a package manager. Create a new Anaconda environment based on Python 3.7 and activate it.
Install PyTorch 0.4.1 as suggested on [PyTorch.org](https://Pytorch.org). Install all other dependencies
```
pip install -r .../smART/requirements.txt
```
Download the pre-trained Deep CNN by [Simonyan and Zisserman, 2014](https://arxiv.org/abs/1409.1556)
from [their site](http://bethgelab.org/media/uploads/pytorch_models/vgg_conv.pth) and move it into `.../smART/data/`.

At this point you should be able to run `simple_test.py` and see very few iterations of NST followed
by ISS sampling a few of the strokes of the library we included with this repo. If you are shown an image of a mountain,
then a weird looking image you will recognize as the style, then a the former rendered in the style of the latter,
then a few strokes placed on a white image, everything worked! 

To run the algorithms on your own target image simply move it into `../smART/data/img/Target_Img/` and call 
`python .../smART/main.py --target-image-filename IMAGE_FILENAME.jpg`. Expect a run of NST to take ~10 minutes on a GPU
using default settings; ISS takes multiple hours.

### Troubleshooting
The default settings use about ~3GB of vRAM on GPUs for ISS, which most modern Desktop GPUs have. If you find PyTorch
telling you to buy more RAM, specify lower values for the parameters `--optimization-img-size`, `--max-batch-size-iss`,
and `--max-batch-size-iss` or simply don't use your CPU (although that will be slow!) by setting the flag `--use_gpu`
accordingly.

If you run into any other issues, please let me know at [ericsteinberger.est@gmail.com](mailto:eric@steinberger-ai.com)!

## How does it work?

### Generating a stroke library to bridge between simulation and actual painting
This is only applicable if you have your own physical smART setup. We used a very simple Genetic Algorithm with
manual selection and no crossover to quickly create and test strokes without having to manually design them. Using this
approach, we are able to have the robot learn multiple strokes for multiple brushes within hours while having
control over how they look by choosing based on visual impressions. We manually postprocess them from a scan
to 160px*160px black-white images, as building a pipeline for that would be unnecessarily complex.

### Neural Style Transfer (NST)
[Neural Style Transfer](https://arxiv.org/pdf/1508.06576.pdf) is an image-stylization algorithm first introduced by Leon
A. Gatys, Alexander S. Ecker and Matthias Bethge in 2015. It uses a pre-trained Deep Convolutional Neural Network to
extract content information from a given "content image" and the style of a given "style image". It then optimizes the
pixel values of a 3rd image, named "painting" in this implementation, using an L-BFGS optimizer. After a few hundred
iterations (<10mins on a GPU) the painting will look like the content image repainted in the style of the style image.

### Iterative Stroke Sampling (ISS)
ISS uses heuristics to create simulated paintings and corresponding motion paths for the robot to execute. It can sample
from the previously created library of strokes that are painted with the different brushes our physical system has.
In this repository, only a few strokes are included. If you don't want to use ISS to generate code for real-life
paintings, you can simply add more artificially created (e.g. via Photoshop) strokes to ISS's library. A simulation
with high-quality settings for a DIN-A3 sized painting takes about 24 hours on an NVIDIA GTX 970; a medium quality
simulation for DIN-A4 (like the examples included in this repository) take around 7 hours.

### ABB RAPID code export
The whole reason we developed and implemented ISS is to get a list of commands for the ABB robot to execute. These are
automatically converted to RAPID code (ABB's robot programming language) and saved to disk in
`.../smART/data/rapid/Paintings` under the name of the image originally handed. In our setup, this code is now passed to
the ABB robot to be executed, which would result in a process like the one you can see in this
[video](https://www.youtube.com/watch?v=AB6-94rLlVI) of the first halve of the painting process of a male face.

## Authors

#### This codebase:
* [**Eric Steinberger**](https://www.linkedin.com/in/ericsteinbergerai/) (TU Vienna)

#### smART team members:
* [**Eric Steinberger**](https://www.linkedin.com/in/ericsteinbergerai/) (TU Vienna) - *Implementation of NST, ISS, stroke GA, and the RAPID translator*
* [**Patrick Pelzmann**](https://www.linkedin.com/in/patrick-pelzmann-8835b1108/) (TU Vienna) - *Color management & brush cleaning machines, hacking the ABB robot*
* [**Benjamin Mörzinger**](https://www.linkedin.com/in/bmoerzin/) (TU Vienna) - *Physical setup, management, supervision*
* [**Manuel Stadler**](https://www.linkedin.com/in/manuel-stadler-85883b151/) (TU Vienna) - *Helped setting up the ABB robot arm*
* [**Alexander Raschendorfer**](https://www.linkedin.com/in/alex-raschendorfer/) (TU Vienna) - *Brush swapping mechanism*
* [**Ralph Oswald**](https://www.linkedin.com/in/ralf-oswald/) (TU Vienna) - *Visualization of robot monitoring logs*

## License
This codebase is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments
 The smART team thanks TU Vienna for supporting and funding this project and HTL Spengergasse for letting Eric
 Steinberger miss lectures to work on it.
