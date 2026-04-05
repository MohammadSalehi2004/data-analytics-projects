# Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# loading the data form csv file into a dataframe
df = pd.read_csv("Data/house_prediction.csv")

# showing some information about the dataset and learning to clean it up from level 1 task 1
print("First 5 rows of the dataset:")
print(df.head(), "\n")

print("Dataset information:")
print(df.info(), "\n")

print("Missing values in each column:")
print(df.isnull().sum(), "\n")

# cleaning up the data frame
df = df.dropna()

# setting varaibles for x and y(similar to machine learning code)
#Features
X = df.drop("price", axis=1)
# Target variable
y = df["price"]               


# training and testing using split dataset so 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# making predictions with the model
y_pred = model.predict(X_test)


# Checking the real values vs the predicted values to find r^2 which will
# tell us how good the model is and then finding mse which will tell how
# wrong the predictions are.
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Model Evaluation:")
print("R² Score:", r2)
print("Mean Squared Error (MSE):", mse, "\n")


# finding the coefficient to showcase how each feature affects the target (prediction)
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("Model Coefficients:")
print(coefficients, "\n")

#creating a scatter plot and saving it at a folder called plots
plt.figure()

#Choosing the x and y for this graph, graph title and axis titles
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")

#saving the image
plt.savefig("Level2/Plots/regression_plot.png")
plt.close()

print("Saved: regression_plot.png")