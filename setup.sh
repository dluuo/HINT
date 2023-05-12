#create hint environment
conda create --name hint python=3.10
source ~/anaconda3/etc/profile.d/conda.sh

conda activate hint

#install dependencies
pip install datasetsforecast statsforecast
pip install git+https://github.com/Nixtla/neuralforecast.git
pip install git+https://github.com/Nixtla/hierarchicalforecast.git

#PROFHiT dependency
pip install properscoring

conda deactivate
