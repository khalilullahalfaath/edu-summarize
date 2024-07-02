@echo off
call conda create -n notesum_env python=3.8 -y
call conda activate notesum_env
call conda install numpy==1.26.4 -y
call conda install pytorch torchvision cpuonly -c pytorch -y
call pip install -r requirements.txt