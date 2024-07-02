# CaDas (catatan cerdas)

## How to install

Note that we've provided batch script so you only have to run it to use the app (only works on Windows)

`
setup.bat
`

then acctivate the env by

`
conda activate notesum_env
`

finally, run the streamlit apps

`
streamlit run main.py
`

Otherwise, you can do these steps also
1. Clone this repository
`git clone https://github.com/khalilullahalfaath/edu-summarize`

2. create a new conda environtment
`
conda create -n {your_desired_env_name} python=3.8
`
ex: cadas_env

3. activate the environtment
`
conda activate {your_env_name}
`

4. IMPORTANT! install numpy and pytorch first using conda
`
conda install numpy==1.26.4
conda install pytorch torchvision cpuonly -c pytorch
`

5. Install other requirements in requirements.txt
`
pip install -r requirements.txt
`

6. Create your env files
    6a. First copy `.env.template`
    6b. Rename it to `.env`
    6c. Write down your azure api key and enpoint correspondingly in the `.env`

7. run the streamlit app
`
streamlit run main.py
`
