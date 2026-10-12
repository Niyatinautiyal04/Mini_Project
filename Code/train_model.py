import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load data with Pandas
data = pd.read_csv('sign_mnist.csv')

# 2. Split into Features (X) and Labels (y)
X = data.iloc[:, 1:].values # All landmark coordinates
y = data.iloc[:, 0].values  # The labels (A, B, C, etc.)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model using Scikit-Learn
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 6. Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to model.pkl")