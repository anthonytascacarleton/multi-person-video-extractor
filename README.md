# Multi-person Video Extractor

## Install

#### (Optional Requirements for GPU Usage)
 - Ubuntu 18.04 x64
 - Nvidia GPU with CUDA 10.2 support

#### Directions
1. Install pip3 and python package dependencies
```
sudo apt-get install python3-pip -y
pip3 install -r requirements.txt
```
2. (Optional Step for GPU Usage) Install cuda 10.2 for Ubuntu 18.04 x64
```
wget -c "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin"
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update -y
sudo apt-get -y install cuda libcupti-dev
```

## Usage

#### Find 10 random multi-person videos in the ```input_folder``` and copy them to the ```output_folder``` while displaying facial detection results with GPU enabled

```
python3 src/multi-person-video-extractor.py \
  --input_folder=./mp4s/ \
  --output_folder=./processing/\
  --num_sample=10 \
  --display_results \
  --use_gpu
```

#### Find 5 random single-person videos in the ```input_folder``` and copy them to the ```output_folder``` without display facial detection results and while using CPU mode

```
python3 src/multi-person-video-extractor.py \
  --input_folder=./mp4s/ \
  --output_folder=./processing/ \
  --num_sample=5 \
  --single_person_mode
```

## Libraries

#### Facial Detection
https://github.com/timesler/facenet-pytorch
