import pandas as pd
import numpy as np

df = pd.read_csv("processed_data.csv")

data = df[['Solar','Wind']].values

LOOKBACK = 24

X = []
Y = []

for i in range(len(data)-LOOKBACK):

    X.append(data[i:i+LOOKBACK])

    Y.append(data[i+LOOKBACK])

X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)

np.save("X.npy",X)
np.save("Y.npy",Y)