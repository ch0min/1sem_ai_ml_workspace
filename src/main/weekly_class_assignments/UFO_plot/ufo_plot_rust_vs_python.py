import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os

# Set OMP_NUM_THREADS to 8
os.environ["OMP_NUM_THREADS"] = "8"

# Python >3.5 is required:
import sys
assert sys.version_info >= (3, 5)

# Sciki-Learn >0.20 is required:
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports:
import numpy as np

# To make this notebook's output stable across runs:
np.random.seed(42)

# To plot pretty figures:
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

# Where to save the figures:
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "unsupervised_learning"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Data
data = {
    'N': [15, 20, 25, 30, 35, 40],
    'Python Time': [0.000099897, 0.001150727, 0.011975455, 0.132887411, 1.451238036, 17.584083676],
    'Rust Time': [0.000003700, 0.000050460, 0.000450210, 0.005051290, 0.056456480, 0.624758590]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Melt the DataFrame to make it suitable for seaborn lineplot
df_melted = df.melt(id_vars='N', var_name='Language', value_name='Time')

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_melted, x='N', y='Time', hue='Language', marker='o')

plt.title('Execution Time Comparison between Python and Rust')
plt.xlabel('N Value')
plt.ylabel('Execution Time (seconds)')
plt.yscale('log')  # Using a logarithmic scale for better visualization
plt.grid(True, which="both", ls="--", linewidth=0.5)
save_fig("rust_vs_python_plot")

plt.show()
