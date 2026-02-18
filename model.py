import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("dataset.csv")
X = data[["feature1", "feature2"]]
y = data["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

accuracy = model.score(X_scaled, y)
print(f"R2 accuracy: {accuracy:.4f}")
