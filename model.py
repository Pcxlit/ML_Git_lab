import pandas as pd
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("dataset.csv")
X = data[["feat1", "feat2"]]
y = data["target"]

model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

predictions = model.predict(X)
accuracy = r2_score(y, predictions)
print(f"R2 accuracy: {accuracy:.4f}")
