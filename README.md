# Overview
Using a grayscale photograph as input, this model hallucinates a possible color version of the photograph. Used the the Caffe framework and LAB color space to implement color mapping algorithms on image inputs to colorize the image.
For more details visit [this article](https://medium.com/@mariya.k2022/deep-learning-techniques-for-image-colorization-a-step-by-step-guide-66c5a4504877) and [video](https://youtu.be/GCN6jIiBhLo). 


# To Use

### 1. Download 
* colorization_deploy_v2.prototxt 

https://github.com/richzhang/colorization/tree/caffe/colorization/models
* pts_in_hull.npy

https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy
* colorization_release_v2.caffemodel

https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1

### 2. Installation

    git clone https://github.com/mariyakhannn/imagecolorizer.git
    cd imagecolorizer
    
### 3. To Run
To run the image colorization script, use the following command:

    python main.py --image <path_to_image>

*Replace <path_to_image> with desired image path.

### Note: .caffemodel file exceeds GitHub's 100MB file size limit. You will need to track this file using Git LFS.

### Credits 

1. https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py
 
2. http://richzhang.github.io/colorization/
 
3. https://github.com/richzhang/colorization/
