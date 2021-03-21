#!/bin/bash

#SBATCH --qos bbpowergpu
#SBATCH --gres gpu:v100:1
#SBATCH --reservation=p9-at-risk 
#SBATCH --ntasks 36
#SBATCH --mem-per-cpu 6800m
#SBATCH --account martynod-optical-modes
#SBATCH --time 7-0:0
#SBATCH --mail-type ALL

set -e

module purge; module load bluebear
module load OpenCV/4.2.0-fosscuda-2019b-Python-3.7.4
module load BEAR-Python-DataScience/2019b-fosscuda-2019b-Python-3.7.4-ppc64le

export VENV_DIR="${HOME}/virtual-environments"
export VENV_PATH="${VENV_DIR}/my-virtual-env-${BB_CPU}"

# Create a master venv directory if necessary
mkdir -p ${VENV_DIR}

# Check if virtual environment exists and create it if not
if [[ ! -d ${VENV_PATH} ]]; then
    virtualenv --system-site-packages ${VENV_PATH}
fi

# Activate the virtual environment
source ${VENV_PATH}/bin/activate

# Perform any required pip installations. For reasons of consistency we would recommend
# that you define the version of the Python module – this will also ensure that if the
# module is already installed in the virtual environment it won't be modified.
#pip3 install --upgrade pip setuptools wheel
#pip3 install scikit-build
#pip3 install cython
#pip3 install tqdm matplotlib scikit-image opencv-python-headless

# Execute your Python scripts
cd ../System/

python3 Main.py -i "ML(SuperpositionGenerator(3, batch_size=8, repeats=8, training_strategy_name='opti_test_strat'), use_multiprocessing=True)" -o "learning_rate" "[round(0.1**n, n) for n in range(8)]"
python3 Main.py -i "ML(SuperpositionGenerator(3, batch_size=8, repeats=8, training_strategy_name='opti_test_strat'), use_multiprocessing=True)" -o "learning_rate" "[round(0.0001 * n, 4) for n in range(1, 9)]"