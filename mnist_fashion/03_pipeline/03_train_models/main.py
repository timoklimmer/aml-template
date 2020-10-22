"""Main script for step: Train Models"""

import argparse
import os
import os.path

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from azureml.core import Run
from tensorflow import keras

print("Training model...")


# --- initialization
print("Initialization...")
# - define and parse script arguments
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--input-dir", type=str, required=True, help="input directory")
parser.add_argument("--epochs", type=int, required=False, default=10, help="number of epochs")
parser.add_argument("--batch-size", type=int, required=False, default=32, help="batch size")
parser.add_argument(
    "--hidden-neurons", type=int, required=False, default=128, help=("number of neurons in the hidden layer")
)
args = parser.parse_args()
input_dir = args.input_dir
epochs = args.epochs
hidden_neurons = args.hidden_neurons
batch_size = args.batch_size
print(f"Epochs         : {epochs}")
print(f"Hidden neurons : {hidden_neurons} ")
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
        keras.layers.Dense(hidden_neurons, activation="relu"),
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
    # pylint: disable=arguments-differ
    def on_epoch_end(self, epoch, log):
        run.log("epoch", epoch)
        run.log("loss", log["loss"])
        run.log("accuracy", log["accuracy"])

    # pylint: enable=arguments-differ


history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, callbacks=[LogRunMetrics()])


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
plt.title(f"Fashion MNIST with Keras ({epochs} EPOCHS)", fontsize=14)
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
        "{} {:2.0f}% ({})".format(
            labels[predicted_label],
            100 * np.max(predictions_array),
            labels[true_label],
        ),
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


# --- upload model
print("Uploading model...")

model_path = f"outputs/model"
neural_net_file_path = f"{model_path}/neural-network.h5"
labels_file_path = f"{model_path}/labels.jsonpickle"

os.mkdir(model_path)

# neural net
model.save(neural_net_file_path)
run.upload_file(neural_net_file_path, neural_net_file_path)

# labels
with open(labels_file_path, "w") as labels_file:
    labels_file.write(jsonpickle.encode(labels))
run.upload_file(labels_file_path, labels_file_path)


# --- Done
print("Done.")
