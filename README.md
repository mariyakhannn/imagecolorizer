# Overview
Using a grayscale photograph as input, this model hallucinates a possible color version of the photograph. Used the the Caffe framework and LAB color space to implement color mapping algorithms on image inputs to colorize the image.

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
    
### 3. To run
To run the image colorization script, use the following command:

    python main.py --image <path_to_image>

*Replace <path_to_image> with desired image path.

### Note: If you plan to push this project to GitHub, be aware that the .caffemodel file exceeds GitHub's 100MB file size limit. You will need to track this file using Git LFS.
