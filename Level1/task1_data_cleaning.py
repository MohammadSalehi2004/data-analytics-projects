#Importing some libraries from the requirements.txt
from pathlib import Path
import pandas as pd

# Create file paths relative to this script so that we can run code from anywhere
# in the folder and avoid worrying about changing directory
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Data"

input_file = DATA_DIR / "stock_prices.csv"
output_file = DATA_DIR / "cleaned_stock_prices.csv"

# Loading dataset and reading it into a pandas dataframe
df = pd.read_csv(input_file)

#confirmation print statement
print("Original dataset loaded successfully.\n")

# Inspecting the dataset to get some information

print("First 5 rows of the dataset:")
print(df.head(), "\n")

print("Dataset shape (rows, columns):")
print(df.shape, "\n")

print("Column names:")
print(df.columns.tolist(), "\n")

print("Data types:")
print(df.dtypes, "\n")

print("Missing values before cleaning:")
print(df.isnull().sum(), "\n")

print("Number of duplicate rows before cleaning:")
print(df.duplicated().sum(), "\n")


# Convert all column names to lowercase and removing spaces to make them neat
df.columns = df.columns.str.lower().str.strip()


# Converting date so its as datetime in the dataframe
# after checking in excel the type for date is made in date form so it 
# has to be consistent and we add an error check so if date is missing its set null
df["date"] = pd.to_datetime(df["date"], errors="coerce")


# Fixing missing values, if the missing value is a number we just replace with median
# A variable is made to show which columns have numbers in it
numeric_columns = ["open", "high", "low", "close", "volume"]

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# This removes any completely duplicated row
df = df.drop_duplicates()

# Double checking every change made
print("Missing values after cleaning:")
print(df.isnull().sum(), "\n")

print("Number of duplicate rows after cleaning:")
print(df.duplicated().sum(), "\n")

print("Updated data types:")
print(df.dtypes, "\n")

print("Cleaned dataset shape:")
print(df.shape, "\n")

# Saving the changes in a new csv file with a new name inside the Data folder
df.to_csv(output_file, index=False)

print(f"Cleaned dataset saved successfully at:\n{output_file}")