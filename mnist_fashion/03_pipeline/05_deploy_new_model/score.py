"""Scoring script for the prediction webservice."""

import io
import traceback

import jsonpickle
import numpy as np
from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from azureml.core import Model
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

neural_network = None
labels = None

# TODO: add schema infos


def init():
    """Is invoked whenever the service is started."""
    global neural_network
    global labels

    # load objects required by run() for inferencing
    model_dir = Model.get_model_path("mnist-fashion")
    # neural model
    neural_network = keras.models.load_model(f"{model_dir}/neural-network.h5")
    # labels
    with open(f"{model_dir}/labels.jsonpickle", "r") as labels_file:
        labels = jsonpickle.decode(labels_file.read())


@rawhttp
def run(request):
    """Is invoked whenever a REST request for prediction is made."""
    try:
        # ensure the user has used a POST request
        if request.method == "POST":
            # prepare data for prediction
            # note: we expect an image of size 28x28 here.
            # TODO: add handling of images that are not 28x28, either resize or deny
            try:
                data = np.asarray(Image.open(io.BytesIO(request.get_data(False)))).reshape(-1, 28, 28)
            except UnidentifiedImageError:
                raise ValueError(
                    "The provided image data could not be read. Ensure that you provide a valid image, eg. in jpeg or "
                    "png format."
                )

            # do prediction
            prediction_confidences = neural_network.predict(data)
            predicted_label_index = np.argmax(prediction_confidences)
            predicted_label = labels[predicted_label_index]
            confidence = prediction_confidences[0][predicted_label_index]

            # return result
            return AMLResponse(
                {"predicted_label": predicted_label, "confidence": str(confidence)}, status_code=200, json_str=True,
            )
        else:
            raise Exception("This service supports POST requests only.")

    except Exception as exception:
        return AMLResponse(
            {"error": repr(exception), "traceback": traceback.format_exc()}, status_code=500, json_str=True,
        )


#
# EXAMPLE: run() method for processing JSON input/output
#
# def run(input_json_string):
#     """Run whenever the service is invoked."""
#     try:
#         # apply the model to the given text
#         sentence = Sentence(json.loads(input_json_string)["text"])
#         classifier.predict(sentence)
#         # return result
#         return {"value": sentence.labels[0].value, "score": sentence.labels[0].score}
#
#     except Exception as e:
#         return {"error": repr(e), "traceback": traceback.format_exc()}
