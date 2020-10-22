"""Main script for step: Extract Data"""

import argparse
import os
import urllib.request

from azureml.core import Run

print("Extracting data...")


# --- initialization
print("Initialization...")
# - define and parse script arguments
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--output-dir", type=str, required=True, help="output directory")
args = parser.parse_args()
output_dir = args.output_dir
# - get run context
run = Run.get_context()
# - ensure that the output directory exists
print("Ensuring that the output directory exists...")
os.makedirs(output_dir, exist_ok=True)


# --- download and save data to output directory
# note: we also could have used AzureML's dataset feature here and register/version the dataset
print("Downloading and saving data...")
files_to_download = {
    "t10k-images-idx3-ubyte.gz": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k"
                                 "-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k"
                                 "-labels-idx1-ubyte.gz",
    "train-images-idx3-ubyte.gz": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train"
                                  "-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train"
                                  "-labels-idx1-ubyte.gz",
}

for file_to_download_name in files_to_download:
    print(f"Downloading file '{file_to_download_name}...")
    file_to_download_path = os.path.join(output_dir, file_to_download_name)
    file_to_download_url = files_to_download[file_to_download_name]
    urllib.request.urlretrieve(file_to_download_url, file_to_download_path)


# --- Done
print("Done.")
