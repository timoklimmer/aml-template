"""Main script for step: Train Model"""

import argparse
import os
import os.path
import time

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from azureml.core import Run
from tensorflow import keras

print("Training model...")


# --- constants
MODEL_NAME = "mnist-fashion"
NN_FILE_NAME = "neural-network.h5"
LABELS_FILE_NAME = "labels.jsonpickle"
# note: AML also supports hyper parameter tuning, see HyperDriveStep for more infos
EPOCHS = 10


# --- initialization
print("Initialization...")
# - define and parse script arguments
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--input_dir", type=str, required=True, help="input directory")
args = parser.parse_args()
input_dir = args.input_dir
# - get run context
run = Run.get_context()


# --- check and log if Tensorflow has access to a GPU
print("Checking if GPU is available...")
is_gpu_available = tf.test.is_gpu_available()
print(f"is_gpu_available: {is_gpu_available}")
run.log("GPU Available", str(is_gpu_available), "Tells if Tensorflow can access a GPU.")


# --- define model
print("Defining model...")
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# --- load data
print("Loading data...")


def load_data(path):
    # train labels
    train_labels = np.load(os.path.join(path, "train-labels-transformed.npz"))["arr_0"]

    # train images
    train_images = np.load(os.path.join(path, "train-images-transformed.npz"))["arr_0"]

    # test labels
    test_labels = np.load(os.path.join(path, "t10k-labels-transformed.npz"))["arr_0"]

    # test images
    test_images = np.load(os.path.join(path, "t10k-images-transformed.npz"))["arr_0"]

    # labels
    labels = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    return train_images, train_labels, test_images, test_labels, labels


train_images, train_labels, test_images, test_labels, labels = load_data(input_dir)


# --- train the model
print("Training model...")


class LogRunMetrics(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log):
        run.log("Epoch", epoch)
        run.log("Loss", log["loss"])
        run.log("Accuracy", log["accuracy"])


history = model.fit(train_images, train_labels, epochs=EPOCHS, callbacks=[LogRunMetrics()])


# --- evaluate the model
print("Evaluating model...")
final_test_loss, final_test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"Final Test Loss     : {final_test_loss}")
print(f"Final Test Accuracy : {final_test_accuracy}")


# --- log metrics and sample predictions
print("Logging metrics and plots...")

# - final test loss and final test accuracy
run.log("Final Test Loss", final_test_loss)
run.log("Final Test Accuracy", final_test_accuracy)

# - loss vs. accuracy image
plt.figure(figsize=(6, 3))
plt.title(f"Fashion MNIST with Keras ({EPOCHS} EPOCHS)", fontsize=14)
plt.plot(history.history["accuracy"], "b-", label="Accuracy", lw=4, alpha=0.5)
plt.plot(history.history["loss"], "r--", label="Loss", lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)
run.log_image("Loss vs. Accuracy", plot=plt)

# - sample predictions
predictions = model.predict(test_images)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"
    plt.xlabel(
        "{} {:2.0f}% ({})".format(labels[predicted_label], 100 * np.max(predictions_array), labels[true_label],),
        color=color,
    )


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
run.log_image("Predictions On Test Set", plot=plt)


# --- save and register model
print("Saving and registering model...")
model_path = f"outputs/{MODEL_NAME}"
os.mkdir(model_path)
model.save(f"{model_path}/{NN_FILE_NAME}")
with open(f"{model_path}/{LABELS_FILE_NAME}", "w") as labels_file:
    labels_file.write(jsonpickle.encode(labels))
time.sleep(5)
run.register_model(
    model_name=MODEL_NAME, model_path=model_path, tags={"Final Test Accuracy": str(final_test_accuracy)},
)


# --- Done
print("Done.")
