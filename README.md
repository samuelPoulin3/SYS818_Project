# SYS818_Project
Classification project of the humain brain by MRI.

The goal of this project is to predict if an image show signs of dementia from
a cross-sectional MRI.

Data were provided by OASIS Cross-Sectional: Principal Investigators: D. Marcus, 
R, Buckner, J, Csernansky J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, 
R01 AG021910, P20 MH071616, U24 RR021382 https://www.oasis-brains.org/

# ------------------------------- STEPS ----------------------------------------
# Download python 3.8.10
https://www.python.org/downloads/release/python-3810/
*restart*

# Set up environement
*Open terminal*
# Check if pip is install:
pip -h

# If nothing show up than install pip 
https://pip.pypa.io/en/latest/installation/

# Once pip is install:
pip install virtualenv
[Python38 path] -m venv [folder to put virtual environment]

Ex: C:\Users\sampo\AppData\Local\Programs\Python\Python38\python -m venv\ C:\Users\sampo\Python\Projects\virtual_env\SYS818_venv

# To activate environment
cd [virtual environment path]
[folder with venv]\Scripts\activate

Ex: cd C:\Users\sampo\Python\Projects\virtual_env
SYS818_venv\Scripts\activate

(Let it activate for now but to deactivate: deactivate)

# If not working because of restriction try
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser

# With virtual environnement activated
# Install packages
cd [project path]\virtual_env
pip install -r requirements.txt

Ex: cd C:\Users\sampo\Python\Projects\SYS818_Project\SYS818\virtual_env
pip install -r requirements.txt

# Choose interpreter (In visual studio)
ctrl+shift+P
Python: select interpreter
browse file
[virtual environment path]\Scripts\python.exe

Ex: C:\Users\sampo\Python\Projects\virtual_env\SYS818_venv\Scripts\python.exe

# Download data


# Change path for data
In src/main.py at line 24-25 change these 2 lines for your own path:
DATA_PATH = "C:/Users/sampo/Python/PycharmProjects/SYS818_Project/Data/subjects"
LABEL_PATH = "C:/Users/sampo/Python/PycharmProjects/SYS818_Project/Data/oasis_cross-sectional.csv"

DATA_PATH contain all the subjects folder
LABEL_PATH is the file oasis_cross-sectional.csv that comes with all the subjects

