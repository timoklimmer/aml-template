@echo off

: Setup a conda environment for using the AzureML SDK for Python from a local dev machine.

SET conda_environment_name=aml_template

: get confirmation from user to avoid something pre-existing might be damaged
echo This installs a conda environment named '%conda_environment_name%'. A pre-existing environment with the same name will be deleted.
choice /C YN /N /M "Are you sure to proceed (y/n)?"
if errorlevel 2 goto :EOF

: remove, create and activate conda environment
call conda activate base
call conda env remove -y -n %conda_environment_name%
call conda create -y -n %conda_environment_name% python=3.6
call conda activate %conda_environment_name%
call conda config --add channels conda-forge

: install some foundational packages via conda and pip
call conda install -y environs autopep8 black yapf isort pylint rope pycodestyle pydocstyle matplotlib plotly tqdm pytest

: install pyspark
: notes: - we should not need hadoop here, so can ignore any hadoop-related messages
:        - to deactivate annoying SET commands shown when the environment is activated, edit the corresponding
:          (de)activate.d scripts.
:        - depending on your machine setup, you might need to add a SET SPARK_LOCAL_IP=127.0.0.1 command to the
:          activate.d script to make pyspark work.
call conda install -y "openjdk>=8,<9"
call conda install -y pyspark

: alternative option: databricks-connect
: run the following commands to install databricks-connect
: notes: - installing databricks-connect and pyspark in parallel is NOT supported, either install pyspark or
:          databricks-connect
:        - make sure you adjust x.x below to match the version of your Azure Databricks cluster
: pip install databricks-connect==x.x., see https://docs.microsoft.com/de-de/azure/databricks/dev-tools/databricks-connect
: databricks-connect configure

: install TensorFlow
: note: if your machine does not have a GPU, use tensorflow instead of tensorflow-gpu
call conda install -y tensorflow-gpu

: install PyTorch
: notes: - adjust as required, depending on your available hardware, see https://pytorch.org
:        - this may lead to conda downgrading TensorFlow. if this is the case, install a different environment or
:          a compatible TensorFlow/PyTorch combination which bases on the same cudatoolkit version.
:        - commented out to avoid conflicts with TensorFlow package from conda
: call conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch

: install Jupyter Lab
call conda install -y nodejs jupyterlab
call jupyter labextension install @jupyterlab/toc --no-build
call jupyter labextension install @aquirdturtle/collapsible_headings --no-build
call conda install -y jupyterlab_code_formatter
call jupyter labextension install @ryantam626/jupyterlab_code_formatter --no-build
call jupyter serverextension enable --py jupyterlab_code_formatter
call python -c "import black; black.CACHE_DIR.mkdir(parents=True, exist_ok=True)"
call jupyter lab build

: install Azure CLI and AzureML SDK for Python
pip install --use-feature=2020-resolver --upgrade azure-cli 
: install AzureML SDK for Python and Azure CLI ML extension
: see https://pypi.org/user/amlpypi for a complete list of available packages, there might be more of interest for you
pip install --use-feature=2020-resolver --upgrade azureml-sdk[accel-models,automl,contrib,explain,interpret,notebooks,services,tensorboard]
pip install --use-feature=2020-resolver --upgrade azureml-defaults azureml-contrib-services azureml-datadrift azureml-explain-model azureml-train-automl azureml-widgets
pip install --use-feature=2020-resolver --upgrade --upgrade-strategy eager azureml-sdk
az extension add --upgrade --name azure-cli-ml

: install our own custom Python packages, if applicable
: TODO: complete as necessary, eg. using pip install -e <package_dir>

: tell user that we're done
echo.
echo Done.
echo.
echo This script did intentionally not install git, VS.Code or PyCharm for you because these do not belong to an
echo individual environment. If you want to use them, ensure you install and configure them manually.
echo Links:
echo - git     : https://git-scm.com/download/win
echo             git config --global user.name "FIRST_NAME LAST_NAME"
echo             git config --global user.email "MY_NAME@example.com"
echo - VS.Code : https://code.visualstudio.com
echo - PyCharm : https://www.jetbrains.com/pycharm