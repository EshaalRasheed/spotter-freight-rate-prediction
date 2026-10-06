# Quick sanity check: confirms libraries import and data files load correctly
import pandas as pd
import lightgbm
import sklearn

print("pandas:", pd.__version__)
print("lightgbm:", lightgbm.__version__)
print("scikit-learn:", sklearn.__version__)

# Load each data file and print its shape (rows, columns)
for name in ["train-test", "validation", "validation-predictions-template", "december-chart-inputs"]:
    df = pd.read_csv(f"data/{name}.csv")
    print(f"{name}: {df.shape}")