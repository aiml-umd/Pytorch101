## 1. Cloning the repository

Run the following command to only clone this folder of the repository
```bash
git clone -n --depth=1 --filter=tree:0 https://github.com/aiml-umd/workshops.git
cd workshops
git sparse-checkout set --no-cone Pytorch101
git checkout
cd Pytorch101
```

## 2. Creating a virtual environment
### Using venv
```bash
python3 -m venv pytorch101_venv
source pytorch101_venv/bin/activate
```
### Using conda
```bash
conda create -n pytorch101 python=3.12
conda activate pytorch101
```

## 3. Installing the requirements
```bash
pip install -r requirements.txt
```

## 4. Running streamlit app
```bash
streamlit run app.py
```