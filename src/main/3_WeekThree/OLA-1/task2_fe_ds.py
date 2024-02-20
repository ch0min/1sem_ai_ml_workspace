import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ************************************************************** #
#                                                                #
#      TASK 2: FEATURE ENGINEERING & DESCRIPTIVE STATISTICS      #
#                                                                #
# ************************************************************** #

# Load dataset
df = pd.read_pickle("../OLA-1/data/interim/task1_data_processed.pkl")

df.info()

# --------------------------------------------------------------
# 1. Feature Engineering
# --------------------------------------------------------------

# Defining age bins and labels
age_bins = [0, 20, 40, 60, np.inf]
age_labels = ["0-20", "21-40", "41-60", "61+"]

df["Age_Group"] = pd.cut(
    df["Age"], bins=age_bins, labels=age_labels, include_lowest=True
)

print(df["Age_Group"].value_counts())
