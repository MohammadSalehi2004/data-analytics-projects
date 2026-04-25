# Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# using this to make the background cleaner as it was used in the other tasks
sns.set_theme()

# CSV file input
input_file = "Data/churn-bigml-80.csv"

# Loading dataset
df = pd.read_csv(input_file)

# showing some information about the dataset
print("First 5 rows of the dataset:")
print(df.head(), "\n")

print("Dataset information:")
print(df.info(), "\n")

print("Missing values in each column:")
print(df.isnull().sum(), "\n")


# cleaning up the data frame
df = df.dropna()

# converting the target column into 0 or 1 since the model needs to read numerical information
df["Churn"] = df["Churn"].astype(int)

# converting categorical columns into numerical columns
df = pd.get_dummies(df, drop_first=True)

# setting variables for x and y
# Features
X = df.drop("Churn", axis=1)

# Target variable
y = df["Churn"]

# training and testing , 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaling the data so numerical values are on a similar scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# creating the machine learning models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}


# testing each model and saving the results
results = []

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(model_name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1, "\n")


# converting results into a dataframe
results_df = pd.DataFrame(results)

print("Model Comparison:")
print(results_df, "\n")


# creating a bar chart to compare model accuracy
plt.figure()

plt.bar(results_df["Model"], results_df["Accuracy"])

plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("Level3/Plots/model_comparison.png")
plt.close()

print("Saved: model_comparison.png")


# hyperparameter tuning using GridSearchCV for Random Forest
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="f1"
)

grid_search.fit(X_train_scaled, y_train)

print("Best Parameters:")
print(grid_search.best_params_, "\n")


# using the best model after tuning
best_model = grid_search.best_estimator_
best_predictions = best_model.predict(X_test_scaled)

print("Best Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, best_predictions))
print("Precision:", precision_score(y_test, best_predictions))
print("Recall:", recall_score(y_test, best_predictions))
print("F1 Score:", f1_score(y_test, best_predictions))