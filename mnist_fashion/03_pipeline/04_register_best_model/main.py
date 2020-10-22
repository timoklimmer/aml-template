"""Main script for step: Register Best Model"""

# pylint: disable=unused-import
import argparse
from azureml.core import Run
from azureml.pipeline.steps import HyperDriveStepRun
from azureml.core import Workspace
from azureml.pipeline.core.run import PipelineRun

# pylint: enable=unused-import

print("Registering best model...")


# --- initialization
print("Initialization...")
# - define and parse script arguments
# TODO: add as required, see other main.py scripts to get a template
# - get run context
run = Run.get_context()


# --- get best model
# to get a pipeline run outside of the pipeline:
# workspace = Workspace(<subscription id>, <resource group>, <workspace name>)
# run = PipelineRun(workspace.experiments[<experiment name>], <run id>)
pipeline_run = PipelineRun(run.experiment, run.parent.id)
train_models_step_run = HyperDriveStepRun(step_run=pipeline_run.find_step_run("Train Models")[0])
best_run = train_models_step_run.get_best_run_by_primary_metric()
final_test_accuracy = best_run.get_metrics("Final Test Accuracy")["Final Test Accuracy"]

# -- validate quality
# TODO: if needed, check if the quality of the model is good enough for a deployment

# --- register model
print("Registering model...")
best_run.register_model(
    model_name="mnist-fashion", model_path="outputs/model", tags={"Final Test Accuracy": str(final_test_accuracy)}
)

# --- Done
print("Done.")
