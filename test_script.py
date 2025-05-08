import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load sample data
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Print basic info
print("✅ All libraries imported successfully!")
print("📊 Dataset shape:", df.shape)
print("📌 First 5 rows:")
print(df.head())

# Plot simple graph
sns.pairplot(df)
plt.show()
