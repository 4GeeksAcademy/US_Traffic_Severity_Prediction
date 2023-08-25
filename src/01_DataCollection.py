# Before executing make sure you have your kaggle.json credential file under
# ~/.kaggle/ folder and do a chmod 600 to kaggle.json for security reasons.
# Alternatively Do a "export KAGGLE_CONFIG_DIR=/workspaces/cesargustavo-Final_Project/.kaggle" and 
#"chmod 600 .kaggle/kaggle.json"
# This is a mandatory step to be able to download the dataset in question.

# Import Packages
import kaggle
import pandas as pd
import os
from IPython.display import display

# Download the dataset
kaggle.api.dataset_download_files('sobhanmoosavi/us-accidents', path="../data/raw", unzip=True)

#Read the csv
dataset = pd.read_csv(r"C:\Users\cesar_v8v3xol\Desktop\Proyecto\data\raw\US_Accidents_March23.csv")

#Show the first 10 lines
display(dataset.head(10))

# Download the dataset
kaggle.api.dataset_download_files('sobhanmoosavi/us-accidents', path="../data/raw", unzip=True)