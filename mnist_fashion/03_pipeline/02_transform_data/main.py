"""Main script for step: Transform Data"""

import argparse
import gzip
import os

import numpy as np
from azureml.core import Run

print("Transforming data...")


# --- initialization
print("Initialization...")
# - define and parse script arguments
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--input-dir", type=str, required=True, help="input directory")
parser.add_argument("--output-dir", type=str, required=True, help="output directory")
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
# - get run context
run = Run.get_context()
# - ensure that the output directory exists
print("Ensuring that the output directory exists...")
os.makedirs(output_dir, exist_ok=True)


# --- load data
def load_gz_data(path):
    # train labels
    with gzip.open(os.path.join(path, "train-labels-idx1-ubyte.gz"), "rb") as label_path:
        train_labels = np.frombuffer(label_path.read(), dtype=np.uint8, offset=8)

    # train images
    with gzip.open(os.path.join(path, "train-images-idx3-ubyte.gz"), "rb") as image_path:
        train_images = np.frombuffer(image_path.read(), dtype=np.uint8, offset=16).reshape(len(train_labels), 28, 28)

    # test labels
    with gzip.open(os.path.join(path, "t10k-labels-idx1-ubyte.gz"), "rb") as label_path:
        test_labels = np.frombuffer(label_path.read(), dtype=np.uint8, offset=8)

    # test images
    with gzip.open(os.path.join(path, "t10k-images-idx3-ubyte.gz"), "rb") as image_path:
        test_images = np.frombuffer(image_path.read(), dtype=np.uint8, offset=16).reshape(len(test_labels), 28, 28)

    return train_images, train_labels, test_images, test_labels


train_images, train_labels, test_images, test_labels = load_gz_data(input_dir)


# --- convert label files to .npz format (for consistency)
print(f"Converting label files to .npz format for consistency...")
np.savez_compressed(os.path.join(output_dir, "train-labels-transformed.npz"), train_labels)
np.savez_compressed(os.path.join(output_dir, "t10k-labels-transformed.npz"), test_labels)


# --- normalize and reshape the images
print(f"Reshaping and normalizing images...")
train_images = train_images / 255.0
np.savez_compressed(os.path.join(output_dir, "train-images-transformed.npz"), train_images)

test_images = test_images / 255.0
np.savez_compressed(os.path.join(output_dir, "t10k-images-transformed.npz"), test_images)


# --- Done
print("Done.")
