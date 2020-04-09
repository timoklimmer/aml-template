"""Main script for step: Deploy model"""

import argparse
import traceback

from azureml.core import Environment, Model, Run, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import AksCompute
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AksWebservice


# temporary workaround function until a bug in environs.read_env(...) is fixed
def find_env_in_parent_directories(env_file_name):
    import os
    import re

    parents = re.split(r"[\\/]", os.path.abspath(env_file_name))[:-1]
    depth = len(parents)
    while depth >= 0:
        path_to_check = "/".join(parents[:depth]) + f"/{env_file_name}"
        if os.path.isfile(path_to_check):
            return path_to_check
        depth -= 1


print("Deploying model as webservice...")


# --- initialization
print("Initialization...")
# - define and parse script arguments
parser = argparse.ArgumentParser(allow_abbrev=False)
args = parser.parse_args()
# - declare other variables for the script
model_name = "mnist-fashion"
service_name = "mnist-fashion-service"
aks_cluster_name = "aks-cluster"
# Token = Azure AD-backed authentication, only Azure AD users with appropriate permissions can use the service.
#         see https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication#web-service-authentica
#         tion for consumption details.
# Keys  = you need to provide one of two keys to get access to the service, keys are shown in ml.azure.com portal.
#         to get the keys programmatically, use service.get_keys()
# None  = No authentication at all, everybody can use the service.
#
# Note: In a later production scenario, you will probably need only one method of authentication.
#       To improve clarity, it's recommended to remove unnecessary code for auth methods not used.
auth_method = "Keys"


# --- ensure that the configured auth_method is in valid range
AUTH_METHOD_TOKEN, AUTH_METHOD_KEYS, AUTH_METHOD_NONE = "Token", "Keys", "None"
VALID_AUTH_METHODS = [AUTH_METHOD_TOKEN, AUTH_METHOD_KEYS, AUTH_METHOD_NONE]
if auth_method not in VALID_AUTH_METHODS:
    raise ValueError(f"The specified auth_method '{auth_method}' is invalid. Valid values are: {VALID_AUTH_METHODS}.")


# --- get run context and workspaces
# - run, default workspace
run = Run.get_context()
if run.id.startswith("OfflineRun"):
    from environs import Env

    env = Env()
    env.read_env(find_env_in_parent_directories("foundation.env"))
    subscription_id = env("SUBSCRIPTION_ID")
    resource_group = env("RESOURCE_GROUP")
    workspace_name = env("WORKSPACE_NAME")
    workspace = Workspace(
        subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name,
    )
else:
    workspace = run.experiment.workspace

# - service-principal authenticated workspace
keyvault = workspace.get_default_keyvault()
deploy_model_sp_tenant_id = keyvault.get_secret(name="DEPLOY-MODEL-SP-TENANT-ID")
deploy_model_sp_app_id = keyvault.get_secret(name="DEPLOY-MODEL-SP-APP-ID")
deploy_model_sp_secret = keyvault.get_secret(name="DEPLOY-MODEL-SP-SECRET")
sp_authenticated_workspace = Workspace(
    subscription_id=workspace.subscription_id,
    resource_group=workspace.resource_group,
    workspace_name=workspace.name,
    auth=ServicePrincipalAuthentication(
        tenant_id=deploy_model_sp_tenant_id,
        service_principal_id=deploy_model_sp_app_id,
        service_principal_password=deploy_model_sp_secret,
    ),
)

# --- get latest model
print("Getting latest model...")
model = next(iter(Model.list(workspace=sp_authenticated_workspace, name=model_name, latest=True)), None,)

# --- prepare deployment
print("Preparing deployment...")
environment = Environment("mnist-fashion-inferencing")
environment.python.conda_dependencies = CondaDependencies.create(
    conda_packages=["numpy", "tensorflow", "jsonpickle", "pillow"], pip_packages=["azureml-defaults",],
)

# --- do deployment
# TODO: put service access keys into KeyVault after deployment
service = None
try:
    service = Model.deploy(
        workspace=sp_authenticated_workspace,
        name=service_name,
        models=[model],
        inference_config=InferenceConfig(entry_script="score.py", environment=environment),
        deployment_config=AksWebservice.deploy_configuration(
            auth_enabled=(auth_method == AUTH_METHOD_KEYS),
            token_auth_enabled=(auth_method == AUTH_METHOD_TOKEN),
            enable_app_insights=True,
        ),
        deployment_target=AksCompute(workspace=sp_authenticated_workspace, name=aks_cluster_name),
        overwrite=True,
    )
    service.wait_for_deployment(show_output=True)
    print("Webservice deployment completed.")
    print(f"State       : {service.state}")
    print(f"Scoring URI : {service.scoring_uri}")
    print(f"Swagger URI : {service.swagger_uri}")
    print(f"Auth method : {auth_method}{' (Azure AD)' if auth_method == AUTH_METHOD_TOKEN else ''}")
    if auth_method == AUTH_METHOD_TOKEN:
        print(
            f"To authenticate against the webservice, ensure the principal accessing the webservice has proper "
            f"permissions and that you pass the right token. Also be aware that tokens expire, so refresh your token "
            f"before it has expired."
        )
    if auth_method == AUTH_METHOD_KEYS:
        print(
            f"For security reasons, the authentication keys for the webservice are not printed here. Get them at "
            f"ml.azure.com."
        )
    if auth_method == AUTH_METHOD_NONE:
        print(
            f"WARNING: The webservice was deployed without any form of authentication. It can be used by ANYONE in the "
            f"internet. Please use a different authentication method for a better secured webservice. You have been "
            f"warned!"
        )

except Exception as exception:
    print("Deployment failed.")
    print(f"Error: {repr(exception)}")
    print(traceback.format_exc())
    if service:
        log_output = service.get_logs()
        if log_output:
            print("Log output:")
            print(log_output)
        else:
            print("No log output from AKS service.")
    run.fail(error_details=exception)


# --- check if the webservice is reachable
# TODO: complete


# --- Done
print("Done.")
