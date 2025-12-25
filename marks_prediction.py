# importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# reading the dataset file
data = pd.read_csv("dataset.csv")

# selecting input columns
X = data[['study_hours', 'attendance', 'internal1', 'internal2']]

# final marks is the value to predict
y = data['final_marks']

# splitting data into training and testing
# 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# creating linear regression model
model = LinearRegression()

# training the model using training data
model.fit(X_train, y_train)

# predicting final marks for test data
predicted_marks = model.predict(X_test)

# printing predicted marks
print("Predicted final marks:")
print(predicted_marks)
