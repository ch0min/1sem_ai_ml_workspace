import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

plt.show()
