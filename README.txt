The required steps to install the correct python environement:

-The code requires python 3.10:
conda create --name yourenvname python=3.10

-To activate the environement:
conda activate yourenvname

-To install pytorch:
conda install pytorch=1.12.0 torchvision==0.13 torchaudio=0.12 cudatoolkit=11.3 -c pytorch

-To install other required packages:
pip install -r requirements.txt
