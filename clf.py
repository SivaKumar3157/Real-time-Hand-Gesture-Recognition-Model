import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
import pickle


df = pd.read_csv("dataset/letters/dataset_aug.csv")
lables = df.iloc[: ,-1].to_numpy()
df = df.iloc[: ,:-1].to_numpy()

print(df)
print(df.shape)
print(lables)

clf = LogisticRegressionCV(cv=10, max_iter=500, n_jobs=-1, random_state=42, scoring='accuracy', verbose=1).fit(df, lables)
filename = 'model_LC.sav'
pickle.dump(clf, open(filename, 'wb'))