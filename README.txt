February 2th 2025:
Fix environement and libraries
Fix dataset download
Fix model


The required steps to install the correct python environement:

-The code requires python 3:
python -m venv emg_env

-To activate the environement:
source emg_env/bin/activate

-To install pytorch:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

-To install other required packages:
pip install -r requirements.txt

-Dataset download authomatically to ./datasets/ninapro8_dataset/

Try a run on a single subject with python main_2models.py
Run training on all subjects separatly using pyton run_all_subjects_ninapro8.py
