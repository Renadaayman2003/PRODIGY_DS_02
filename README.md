# Titanic Dataset Analysis

## Introduction

This project demonstrates the steps to analyze the Titanic dataset using Python. The analysis involves downloading the dataset from Kaggle, performing data cleaning, exploratory data analysis (EDA), and visualizing the results.

## Step 1: Install and Set up Kaggle API

Before starting, ensure you have access to the Kaggle API and your API key (`kaggle.json`) is ready. Follow these commands to install and configure the API:

```bash
!pip install kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

## Step 2: Download Titanic Dataset

Use the Kaggle API to download the Titanic dataset:

```bash
!kaggle datasets download -d brendan45774/test-file
```

Once downloaded, extract the contents of the zip file:

```python
import zipfile
with zipfile.ZipFile('test-file.zip', 'r') as zip_ref:
    zip_ref.extractall('titanic_data')
```

## Step 3: Load the Dataset

List the contents of the extracted folder and load the dataset into a Pandas DataFrame:

```python
import os
print(os.listdir('titanic_data'))

import pandas as pd
df = pd.read_csv('titanic_data/tested.csv')
print(df.head())
```

## Step 4: Data Inspection

To understand the data, inspect the structure and look for missing values:

```python
print(df.info())         # Data types and missing values
print(df.describe())     # Summary statistics
print(df.isnull().sum()) # Count of missing values in each column
```

## Step 5: Data Cleaning

### 1. Handling Missing Values:

- Fill missing values in the 'Age' column with the median value.
- Replace missing 'Embarked' values with the most common value.
- Drop the 'Cabin' column due to the large number of missing entries.

```python
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)
```

### 2. Handling Duplicates:

Remove duplicate entries in the dataset:

```python
df.drop_duplicates(inplace=True)
```

### 3. Data Type Conversion:

Convert categorical variables into numerical representations:

```python
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
```

## Step 6: Exploratory Data Analysis (EDA)

### 1. Age Distribution:

Visualize the distribution of ages in the dataset:

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.show()
```

### 2. Survival Rate:

Visualize the number of survivors and non-survivors:

```python
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()
```

### 3. Bivariate Analysis:

#### Survival by Gender:

Examine how survival rates differ between males and females:

```python
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Gender')
plt.show()
```

#### Survival by Passenger Class (Pclass):

Analyze survival rates by passenger class:

```python
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival by Passenger Class')
plt.show()
```

### 4. Multivariate Analysis

#### Correlation Matrix:

Generate a heatmap to visualize correlations between numeric variables:

```python
numeric_df = df.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

## Conclusion

This project walks through setting up the Kaggle API, loading the Titanic dataset, performing basic data cleaning, and conducting exploratory data analysis using visualizations. Further steps could include building predictive models to assess survival probability.
