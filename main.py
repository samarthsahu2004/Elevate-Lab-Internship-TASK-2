# ==============================================================================
# TITANIC DATASET: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. SETUP AND DATA LOADING
# ------------------------------------------------------------------------------
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import kagglehub

# Set plot style for better aesthetics
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# --- Download and Load Data ---
print("Downloading and loading the dataset...")
# Download the dataset
path = kagglehub.dataset_download("yasserh/titanic-dataset")

# The downloaded folder contains 'Titanic.csv'. Let's construct the full path.
file_path = f"{path}/Titanic.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!\n")


# ------------------------------------------------------------------------------
# 2. INITIAL DATA INSPECTION
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("2. INITIAL DATA INSPECTION")
print("="*50 + "\n")

# Display the first 5 rows of the dataframe
print("--- First 5 rows of the dataset ---")
print(df.head())

print("\n--- Dataframe Information ---")
# Get a concise summary of the dataframe
df.info()

print("\n--- Summary Statistics for Numerical Features ---")
# Generate summary statistics for numerical features
print(df.describe())

print("\n--- Summary Statistics for Categorical Features ---")
# Generate summary statistics for categorical features
print(df.describe(include=['object']))


# ------------------------------------------------------------------------------
# 3. UNIVARIATE ANALYSIS (ANALYZING SINGLE VARIABLES)
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("3. UNIVARIATE ANALYSIS (GENERATING PLOTS)")
print("="*50 + "\n")

# --- Numeric Features: Histograms and Boxplots ---
print("Displaying plots for numerical features distribution...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Distribution of Numerical Features', fontsize=20)

# Age distribution
sns.histplot(df['Age'].dropna(), kde=True, ax=axes[0, 0], bins=30)
axes[0, 0].set_title('Age Distribution')

# Fare distribution
sns.histplot(df['Fare'], kde=True, ax=axes[0, 1], bins=40)
axes[0, 1].set_title('Fare Distribution')

# SibSp (Siblings/Spouses) distribution
sns.histplot(df['SibSp'], ax=axes[1, 0], discrete=True)
axes[1, 0].set_title('Siblings/Spouses Aboard')

# Parch (Parents/Children) distribution
sns.histplot(df['Parch'], ax=axes[1, 1], discrete=True)
axes[1, 1].set_title('Parents/Children Aboard')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("Displaying boxplots for outlier detection...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Boxplots for Outlier Detection', fontsize=20)

# Age boxplot
sns.boxplot(x=df['Age'], ax=axes[0])
axes[0].set_title('Age Boxplot')

# Fare boxplot
sns.boxplot(x=df['Fare'], ax=axes[1])
axes[1].set_title('Fare Boxplot')
plt.show()

# --- Categorical Features: Count Plots (Interactive) ---
print("Displaying interactive plots for categorical features...")
# Note: Plotly plots will open in your web browser or viewer.
fig = px.bar(df, x='Survived', title='Survival Count (0 = No, 1 = Yes)', labels={'Survived':'Survival Status'})
fig.show()

fig = px.bar(df, x='Pclass', title='Passenger Class Distribution', labels={'Pclass':'Passenger Class'})
fig.show()

fig = px.bar(df, x='Sex', title='Gender Distribution', labels={'Sex':'Gender'})
fig.show()

fig = px.bar(df, x='Embarked', title='Port of Embarkation', labels={'Embarked':'Port'})
fig.show()


# ------------------------------------------------------------------------------
# 4. BIVARIATE & MULTIVARIATE ANALYSIS (ANALYZING RELATIONSHIPS)
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("4. BIVARIATE & MULTIVARIATE ANALYSIS (GENERATING PLOTS)")
print("="*50 + "\n")

# --- Correlation Matrix ---
print("Displaying correlation heatmap...")
# Select only numeric columns for correlation matrix
numeric_cols = df.select_dtypes(include=np.number)

# Calculate the correlation matrix
corr_matrix = numeric_cols.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# --- Feature Relationships with Survival ---
print("Displaying plots showing relationships with survival...")
# Survival rate by Pclass
sns.catplot(x='Pclass', col='Survived', data=df, kind='count', height=6, aspect=0.7)
plt.suptitle('Survival Count by Passenger Class', y=1.02)
plt.show()

# Survival rate by Sex
sns.catplot(x='Sex', col='Survived', data=df, kind='count', height=6, aspect=0.7)
plt.suptitle('Survival Count by Gender', y=1.02)
plt.show()

# Survival rate by Pclass and Sex
sns.catplot(x='Pclass', hue='Sex', col='Survived', data=df, kind='count', height=6, aspect=0.8)
plt.suptitle('Survival by Pclass and Sex', y=1.02)
plt.show()

# Age distribution of Survived vs. Deceased
g = sns.FacetGrid(df, col='Survived', height=6)
g.map(sns.histplot, 'Age', bins=25, kde=True)
plt.suptitle('Age Distribution by Survival Status', y=1.02)
plt.show()

# --- Pairplot ---
print("Displaying pairplot of key features...")
# Create a pairplot, colored by survival status
# We'll use a subset of columns for clarity
pairplot_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
sns.pairplot(df[pairplot_cols].dropna(), hue='Survived', palette='husl')
plt.suptitle('Pairplot of Key Features by Survival Status', y=1.02)
plt.show()

print("\nEDA script finished.")
# ==============================================================================
# END OF FILE
# ==============================================================================
