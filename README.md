## 1. Cloning the repository

Run the following command to clone the repository
```bash
git clone https://github.com/aiml-umd/Pytorch101.git
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