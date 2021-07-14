"""Creates and deploys the model training and deployment pipeline."""

# pylint: disable=unused-import
import argparse

from azureml.core import Experiment, Workspace
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import CondaDependencies, RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData, PipelineEndpoint
from azureml.pipeline.core.schedule import Schedule, ScheduleRecurrence
from azureml.pipeline.steps import EstimatorStep, HyperDriveStep, PythonScriptStep
from azureml.train.dnn import TensorFlow
from azureml.train.hyperdrive import (
    BanditPolicy,
    BayesianParameterSampling,
    HyperDriveConfig,
    PrimaryMetricGoal,
    RandomParameterSampling,
    choice,
)
from environs import Env

# pylint: enable=unused-import

# --- define and parse script arguments
parser = argparse.ArgumentParser(allow_abbrev=False)
trigger_after_publish_parser = parser.add_mutually_exclusive_group(required=True)
trigger_after_publish_parser.add_argument(
    "--trigger-after-publish",
    dest="trigger_after_publish",
    action="store_true",
    help="Triggers the pipeline after it has been published.",
)
trigger_after_publish_parser.add_argument(
    "--no-trigger-after-publish",
    dest="trigger_after_publish",
    action="store_false",
    help="Publishes the pipeline but does not trigger it.",
)

schedule_parser = parser.add_mutually_exclusive_group(required=True)
schedule_parser.add_argument(
    "--schedule",
    dest="schedule",
    action="store_true",
    help="Attaches the schedule to the pipeline.",
)
schedule_parser.add_argument(
    "--no-schedule",
    dest="schedule",
    action="store_false",
    help="Skips attaching the schedule to the pipeline.",
)
args = parser.parse_args()
trigger_after_publish = args.trigger_after_publish
schedule = args.schedule


# --- load configuration
print("Loading configuration...")
env = Env(expand_vars=True)
env.read_env("foundation.env")

azure_subscription_id = env("AZURE_SUBSCRIPTION_ID")
resource_group = env("RESOURCE_GROUP")
workspace_name = env("WORKSPACE_NAME")
workspace_region = env("WORKSPACE_REGION")
gpu_cluster_name = env("GPU_BATCH_CLUSTER_NAME")


# --- get workspace, compute target, run config
print("Getting workspace and compute target...")
workspace = Workspace(
    subscription_id=azure_subscription_id,
    resource_group=resource_group,
    workspace_name=workspace_name,
)
compute_target = ComputeTarget(workspace=workspace, name=gpu_cluster_name)
run_config = RunConfiguration(
    conda_dependencies=CondaDependencies.create(
        conda_packages=[],
        # notes: - see run_config.environment.add_private_pip_wheel() to use your own private packages,
        #        - you can also reference curated or custom environments here for simplification,
        #          see https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-your-first-pipeline for
        #          more details
        pip_packages=["azureml-defaults", "azureml-pipeline-steps"],
    )
)
run_config.environment.docker.enabled = True
# recommendation: use a fixed image in production to avoid sudden surprises
#                 check DEFAULT_CPU_IMAGE or DEFAULT_GPU_IMAGE for the newest image
# from azureml.core.runconfig import DEFAULT_CPU_IMAGE, DEFAULT_GPU_IMAGE
run_config.environment.docker.base_image = (
    "mcr.microsoft.com/azureml/intelmpi2018.3-cuda10.0-cudnn7-ubuntu16.04:20200821.v1"
)
run_config.environment.spark.precache_packages = False

# --- define pipeline steps
# note: use <pipline step>.run_after() if there is a dependency but no input/output connection
#       without these relations set up, all steps will run in parallel by default
print("Defining pipeline steps...")

# - Extract Data
extracted_data_dir = PipelineData(
    "extracted_data",
    is_directory=True,
)
extract_data_step = PythonScriptStep(
    name="Extract Data",
    script_name="main.py",
    source_directory="01_extract_data",
    compute_target=compute_target,
    runconfig=run_config,
    outputs=[extracted_data_dir],
    arguments=["--output-dir", extracted_data_dir],
    allow_reuse=False,
)

# - Transform Data
transformed_data_dir = PipelineData(
    name="transformed_data",
    is_directory=True,
)
transform_data_step = PythonScriptStep(
    name="Transform Data",
    script_name="main.py",
    source_directory="02_transform_data",
    compute_target=compute_target,
    runconfig=run_config,
    inputs=[extracted_data_dir],
    outputs=[transformed_data_dir],
    arguments=["--input-dir", extracted_data_dir, "--output-dir", transformed_data_dir],
    allow_reuse=False,
)

# - Train Models

# # option 1 - no hyperparameter optimization, single model without hyperparameter tuning
# more infos on estimators at:
# https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.estimator.estimator.
#
# The example below uses the pre-defined TensorFlow estimator but there is also other estimators and the option to build
# your own estimator.
#
# train_model_step = EstimatorStep(
#     name="Train Model",
#     estimator=TensorFlow(
#         entry_script="main.py",
#         source_directory="03_train_models",
#         compute_target=compute_target,
#         framework_version="2.0",
#         conda_packages=[],
#         pip_packages=["matplotlib"],
#         use_gpu=True,
#     ),
#     compute_target=compute_target,
#     inputs=[transformed_data_dir],
#     estimator_entry_script_arguments=["--input-dir", transformed_data_dir],
#     allow_reuse=False,
# )

# option 2 - hyperparameter optimization using HyperDrive
# - see https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters for more details and further
#   options wrt. hyperparameter selection strategies and termination policies.
# - also check if the settings below are valid in case you use this in a production context
train_models_step = HyperDriveStep(
    name="Train Models",
    hyperdrive_config=HyperDriveConfig(
        estimator=TensorFlow(
            entry_script="main.py",
            source_directory="03_train_models",
            compute_target=compute_target,
            framework_version="2.2",
            conda_packages=[],
            pip_packages=["matplotlib"],
            use_gpu=True,
        ),
        hyperparameter_sampling=RandomParameterSampling(
            {
                "--epochs": choice(10, 25, 50, 100),
                "--hidden-neurons": choice(10, 50, 200, 300, 500),
                "--batch-size": choice(32, 64, 128, 256),
            }
        ),
        policy=BanditPolicy(evaluation_interval=3, slack_amount=0.05),
        primary_metric_name="accuracy",
        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
        max_total_runs=20,
        max_concurrent_runs=4,
        max_duration_minutes=120,
    ),
    inputs=[transformed_data_dir],
    estimator_entry_script_arguments=["--input-dir", transformed_data_dir],
)

# - Register Best Model
register_best_model_step = PythonScriptStep(
    name="Register Best Model",
    script_name="main.py",
    source_directory="04_register_best_model",
    compute_target=compute_target,
    runconfig=run_config,
    allow_reuse=False,
)
register_best_model_step.run_after(train_models_step)

# - Deploy New Model
deploy_new_model_step = PythonScriptStep(
    name="Deploy New Model",
    script_name="main.py",
    source_directory="05_deploy_new_model",
    compute_target=compute_target,
    runconfig=run_config,
    allow_reuse=False,
)
deploy_new_model_step.run_after(register_best_model_step)

# --- assemble and publish publishing pipeline
# note: see here for infos on how to schedule the pipeline
#       https://github.com/Azure/MachineLearningNotebooks/blob/fe8fcd4b480dab7fee9fa32a354132e4df25db8a/how-to-use-azure
#       ml/machine-learning-pipelines/intro-to-pipelines/aml-pipelines-setup-schedule-for-a-published-pipeline.ipynb
print("Assembling and publishing pipeline...")
pipeline_name = "MNIST-Fashion-Pipeline"
pipeline_description = "Just a simple sample pipeline."
pipeline_steps = [
    extract_data_step,
    transform_data_step,
    train_models_step,
    register_best_model_step,
    deploy_new_model_step,
]
pipeline = Pipeline(
    workspace=workspace,
    steps=pipeline_steps,
    description=pipeline_description,
)
pipeline.validate()

published_pipeline = pipeline.publish(
    # name=pipeline_name, description=pipeline_description, version={...some version...}
    name=pipeline_name,
    description=pipeline_description,
)
print(f"Newly published pipeline id: {published_pipeline.id}")

try:
    pipeline_endpoint = PipelineEndpoint.get(workspace=workspace, name=pipeline_name)
    pipeline_endpoint.add(published_pipeline)
except:
    pipeline_endpoint = PipelineEndpoint.publish(
        workspace=workspace,
        name=pipeline_name,
        pipeline=published_pipeline,
        description=f"Pipeline Endpoint for {pipeline_name}",
    )

pipeline_endpoint.set_default(published_pipeline)

# TODO: cleanup older pipeline endpoints(?)


# --- add a schedule for the pipeline (if told to do so)
# note: this is a sample schedule which runs time-based.
#       there is also the option to trigger the pipeline based on changes.
#       details at https://github.com/Azure/MachineLearningNotebooks/blob/4e7b3784d50e81c313c62bcdf9a330194153d9cd/how-t
#       o-use-azureml/machine-learning-pipelines/intro-to-pipelines/aml-pipelines-setup-schedule-for-a-published-pipelin
#       e.ipynb
if schedule:
    recurrence = ScheduleRecurrence(frequency="Day", interval=2, hours=[22], minutes=[30])
    schedule = Schedule.create(
        workspace=workspace,
        name="Every-Other-Day-At-10-30-PM",
        pipeline_id=published_pipeline.id,
        experiment_name=pipeline_name,
        recurrence=recurrence,
        wait_for_provisioning=True,
        description="A sample schedule which runs every other day at 10:30pm.",
    )


# --- trigger pipeline endpoint if we have been told to do so
if trigger_after_publish:
    print(f"Triggering pipeline endpoint '{pipeline_name}' (as configured)...")
    pipeline_run = Experiment(workspace, pipeline_name).submit(pipeline_endpoint)
    pipeline_run.wait_for_completion()


# --- Done
print("Done.")
