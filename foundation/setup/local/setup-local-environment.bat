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

: install Jupyter Lab
call conda install -y nodejs jupyterlab
call jupyter labextension install @jupyterlab/toc --no-build
call jupyter labextension install @aquirdturtle/collapsible_headings --no-build
call conda install -y jupyterlab_code_formatter
call jupyter labextension install @ryantam626/jupyterlab_code_formatter --no-build
call jupyter serverextension enable --py jupyterlab_code_formatter
call python -c "import black; black.CACHE_DIR.mkdir(parents=True, exist_ok=True)"
call jupyter lab build

: install Azure CLI, AzureML SDK + Azure CLI ML extension
: Azure CLI
call pip install --use-feature=2020-resolver azure-cli
: AzureML SDK for Python
: - There is more AzureML packages that might be of interest for you.
:   See https://docs.microsoft.com/de-de/python/api/overview/azure/ml/install (incl. menu on the left) and
:   https://pypi.org/user/amlpypi for a complete list of available packages.
: - There is also an R SDK (not coverered here), see https://github.com/Azure/azureml-sdk-for-r
call pip install --use-feature=2020-resolver wrapt azureml-sdk
: TODO: install additional packages as needed
: Azure CLI ML extension
call az extension add --upgrade --name azure-cli-ml

: install CUDA toolkit for TensorFlow and PyTorch
: note: intentionally using 10.1 as this version is compatible with both TF 2.2. and PyTorch 1.6.
:       change to newer versions as needed and compatibility is given.
call conda install -y cudatoolkit=10.1

: install TensorFlow
: note: if your machine does not have a GPU, use package "tensorflow" instead of "tensorflow-gpu"
call pip install --use-feature=2020-resolver tensorflow-gpu==2.2

: install PyTorch
: notes: - adjust as required, depending on your available hardware and other packages used, see https://pytorch.org
:        - this may lead to conda downgrading TensorFlow. if this is the case, install a different environment or
:          a compatible TensorFlow/PyTorch combination which bases on the same cudatoolkit version.
call conda install -y pytorch==1.6.0 torchvision==0.7.0 -c pytorch

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