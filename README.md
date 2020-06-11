# Image_Captioning
This project conatins file from Udacity Computer Vision Nanodegree 

In this project we combine CNN and RNN as decoder and encoder respectively, to produce caption for images from the [COCO Dataset - Common Objects in Context](http://cocodataset.org/). 
<p align="center"> <img src="images/encoder-decoder.png" align="middle" alt="drawing" width="900px"> </p> 

## Dataset
<p align="center"> <img src="images/coco-examples.jpg" align="middle" alt="drawing" width="900px"> </p> 

To set up the COCOAPI to use the dataset, 
follow the instruction in this [readme file](https://github.com/udacity/CVND---Image-Captioning-Project/)



## Project Structure
The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order:

__Notebook 0__ : Microsoft Common Objects in COntext (MS COCO) dataset;

__Notebook 1__ : Load and pre-process data from the COCO dataset;

__Notebook 2__ : Training the CNN-RNN Model;

__Notebook 3__ : Load trained model and generate predictions.

## Installation
```sh
$ git clone https://github.com/kenkai/Image_Captioning.git
$ pip3 install -r requirements.txt
```
## References
[Microsoft COCO](https://arxiv.org/pdf/1405.0312.pdf), [arXiv:1411.4555v2 [cs.CV] 20 Apr 2015](https://arxiv.org/pdf/1411.4555.pdf) </li>
and [arXiv:1502.03044v3 [cs.LG] 19 Apr 2016](https://arxiv.org/pdf/1502.03044.pdf)

## Licence
This project is licensed under the terms of the [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
