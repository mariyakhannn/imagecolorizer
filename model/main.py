import numpy as np
import argparse
import cv2
import os

# show where paths to model files are
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, 'model')
PROTOTXT = os.path.join(model_dir, "colorization_deploy_v2.prototxt")
MODEL = os.path.join(model_dir, "colorization_release_v2.caffemodel")
POINTS = os.path.join(model_dir, "pts_in_hull.npy")

# argparser
parser = argparse.ArgumentParser(description='Colorize an image using a pre-trained model')
parser.add_argument('--image', type=str, required=True, help='Path to the image to colorize')
parser.add_argument('--output', type=str, required=True, help='Path to save the colorized image')
args = parser.parse_args()

# load the model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# load cluster centers for ab values
pts = np.load(POINTS).transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype='float32')] 

# load input image
img = cv2.imread(args.image)
if img is None: 
    raise ValueError(f"Error: unable to open image file {args.image}")

img = img.astype(np.float32) / 255.0

# convert the image to grayscale and back to BGR to match model input
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
img_resized = cv2.resize(gray, (224, 224))

blob = cv2.dnn.blobFromImage(img_resized, 1.0, (224, 224), (50, 50, 50))
net.setInput(blob)

output = net.forward()

# find output, resize to the original image dimensions, and convert to uint8
output_ab = output[0, :, :, :].transpose((1, 2, 0))
output_ab = cv2.resize(output_ab, (img.shape[1], img.shape[0]))

img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
L_channel = img_lab[:, :, 0]
colorized_img = np.concatenate((L_channel[:, :, np.newaxis], output_ab), axis=2)
colorized_img = cv2.cvtColor(colorized_img, cv2.COLOR_Lab2BGR)

colorized_img = np.clip(colorized_img, 0, 255).astype("uint8")

cv2.imwrite(args.output, colorized_img)
print(f"Colorized image saved to {args.output}")
cv2.imshow("Colorized Image", colorized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()