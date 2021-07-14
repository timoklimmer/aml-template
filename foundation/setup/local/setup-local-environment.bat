@echo off

: Setup a conda environment for using the AzureML SDK for Python from a local dev machine.
: Notes: The script installs only a subset of available packages for AML.
: - There is more AzureML packages that might be of interest for you.
:   See https://docs.microsoft.com/en-us/python/api/overview/azure/ml/install (incl. menu on the left) and
:   https://pypi.org/user/amlpypi for a complete list of available packages.
: - There is also an R SDK (not coverered here), see https://github.com/Azure/azureml-sdk-for-r

SET conda_environment_name=aml_template

: get confirmation from user to avoid something pre-existing might be damaged
echo This installs a conda environment named '%conda_environment_name%'. A pre-existing environment with the same name will be deleted. The script will also remove a pre-existing azure-cli-ml extension for the az command.
choice /C YN /N /M "Are you sure to proceed (y/n)?"
if errorlevel 2 goto :EOF

: remove, create and activate conda environment
call conda activate base
call conda env remove -y -n %conda_environment_name%
call conda create -y -n %conda_environment_name% python=3.8
call conda activate %conda_environment_name%
call conda config --add channels conda-forge

: install some foundational packages via conda and pip
call conda install -y environs autopep8 black yapf isort pylint rope pycodestyle pydocstyle matplotlib plotly tqdm pytest

: install Jupyter Lab
call conda install -y nodejs jupyterlab
call jupyter labextension install @jupyterlab/toc --no-build
call jupyter labextension install @aquirdturtle/collapsible_headings --no-build
call conda install -y jupyterlab_code_formatter
call jupyter labextension install @ryantam626/jupyterlab_code_formatter --no-build
call jupyter serverextension enable --py jupyterlab_code_formatter
call jupyter lab build

: install Azure CLI, AzureML SDK + Azure CLI ML extension
: Azure CLI
call pip install azure-cli
: AzureML SDK for Python
call pip install azureml-core azureml-sdk azureml-mlflow azureml-monitoring azureml-datadrift
: Azure CLI ML extension
: remove the old azure-cli-ml extension if existing
call az extension remove -n azure-cli-ml
: install the new ml extension
call az extension add -n ml

: install CUDA toolkit for TensorFlow and PyTorch
call conda install -y cudatoolkit=11.1 cudnn

: install TensorFlow
: note: if your machine does not have a GPU, use package "tensorflow" instead of "tensorflow-gpu"
call pip install tensorflow-gpu==2.5

: install PyTorch
: notes: - adjust as required, depending on your available hardware and other packages used, see https://pytorch.org
:        - this may lead to conda downgrading TensorFlow. if this is the case, install a different environment or
:          a compatible TensorFlow/PyTorch combination which bases on the same cudatoolkit version.
call conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch

: tell user that we're done
echo.
echo Done.
echo.
echo This script did intentionally not install git, VS.Code or PyCharm for you because these do not belong to an
echo individual environment. If you want to use them, ensure you install and configure them manually.
echo Links:
echo - git     : https://git-scm.com/download/win
echo             git config --global user.name "FIRST_NAME LAST_NAME"
echo             git config --global user.email "...@users.noreply.github.com (adjust as needed)"
echo - VS.Code : https://code.visualstudio.com
echo - PyCharm : https://www.jetbrains.com/pycharm