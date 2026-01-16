import pandas as pd

data = {
    "Model": ["SVM (TF-IDF)", "CNN-LSTM"],
    "Accuracy": [0.88, 0.86],
    "Precision": [0.88, 0.86],
    "Recall": [0.87, 0.85],
    "F1-Score": [0.87, 0.85]
}

df = pd.DataFrame(data)
print(df)
