from django.db import models

# Create your models here.
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle

# Load dataset
data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.DataFrame(data.target, columns=["MEDV"])

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(df, target, test_size = 0.2)

# Train the model
l_regression = LinearRegression()
l_regression.fit(X_train, Y_train)

# Save the model
with open('predictor/model.pkl', 'wb') as f:
    pickle.dump(l_regression, f)