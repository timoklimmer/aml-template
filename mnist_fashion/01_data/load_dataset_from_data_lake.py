"""
Register a FileDataset and download the contained files.

Notes: - Besides the FileDataset, there is also TabularDatasets which are better suited for tabular data.
         See https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets for more details.

       - Besides downloading, datasets can also be mounted into the file system (Unix). Depending on the data volumes,
         a mount might be the better option.
"""

import os
from azureml.core import Workspace, Datastore, Dataset
from environs import Env

# get workspace and datastore
env = Env()
env.read_env("foundation.env")
workspace = Workspace(env("AZURE_SUBSCRIPTION_ID"), env("RESOURCE_GROUP"), env("WORKSPACE_NAME"))
datastore = Datastore.get(workspace, env("SOME_EXTERNAL_ADLS_GEN2_DATASTORE_NAME"))

# create a FileDataset pointing to the existing files in folder mnist-fashion on the data lake account
datastore_paths = [(datastore, "mnist-fashion")]
dataset = Dataset.File.from_files(path=datastore_paths)

# register the dataset to enable later reuse
dataset.register(workspace, "mnist-fashion", "Sample dataset", create_new_version=True)

# get the dataset (for the case we have no dataset instance at this point)
dataset = Dataset.get_by_name(workspace=workspace, name="mnist-fashion")

# download the files to local folder
# note: this is only one of multiple options. for instance, on Unix machines, the dataset can also be mounted
directory_of_this_script = os.path.dirname(os.path.realpath(__file__))
dataset.download(directory_of_this_script)
