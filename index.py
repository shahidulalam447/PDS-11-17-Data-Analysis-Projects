import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Project 1: Data Manipulation, Handling Missing Data, Data Visualization

data_url = "data/titanic.csv"
df = pd.read_csv(data_url)
print(df.head())

# Step 2: Find feature names with null values
null_columns = df.columns[df.isnull().any()].tolist()
print("Columns with null values:", null_columns)

# Step 3: Fill missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Cabin'].fillna('Unknown', inplace=True)

# Step 4 & 5: Bar plot of Survived vs Dead by Gender and Pclass
plt.figure(figsize=(10, 5))
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Count by Gender')
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival Count by Passenger Class')
plt.show()

# Step 6: Create AgeClass column
def age_class(age):
    if age <= 16:
        return 0
    elif age <= 26:
        return 1
    elif age <= 36:
        return 2
    elif age <= 62:
        return 3
    else:
        return 4

df['AgeClass'] = df['Age'].apply(age_class)

# Step 7: Drop Age column
df.drop(columns=['Age'], inplace=True)

# Step 8: Bar plot of Survived vs Dead by AgeClass
plt.figure(figsize=(10, 5))
sns.countplot(x='Survived', hue='AgeClass', data=df)
plt.title('Survival Count by Age Class')
plt.show()





# Project 2: Dataset Merging, Data Manipulation, K-Means Clustering

# Step 1 & 2: Create term-test result CSVs
term_test_1 = pd.DataFrame({
    'Registration Number': range(1, 51),
    'Name': [f'Student {i}' for i in range(1, 51)],
    'TT-1 Marks': np.random.randint(40, 100, 50)
})
term_test_1.to_csv('term-test-1-result.csv', index=False)

term_test_2 = term_test_1.copy()
term_test_2['TT-2 Marks'] = np.random.randint(40, 100, 50)
term_test_2.drop(columns=['TT-1 Marks'], inplace=True)
term_test_2.to_csv('term-test-2-result.csv', index=False)

# Step 3: Load both files and merge on Registration Number
tt1 = pd.read_csv('term-test-1-result.csv')
tt2 = pd.read_csv('term-test-2-result.csv')
merged_df = pd.merge(tt1, tt2, on=['Registration Number', 'Name'])

# Step 4: Compute best and average marks
merged_df['Best Marks'] = merged_df[['TT-1 Marks', 'TT-2 Marks']].max(axis=1)
merged_df['Average Marks'] = merged_df[['TT-1 Marks', 'TT-2 Marks']].mean(axis=1)

# Step 5: Drop TT-1 and TT-2 Marks
merged_df.drop(columns=['TT-1 Marks', 'TT-2 Marks'], inplace=True)

# Step 6: Save to final CSV
merged_df.to_csv('final-term-test-result.csv', index=False)

# Step 7: K-Means Clustering and Visualization
X = merged_df[['Average Marks']]
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
merged_df['Cluster'] = kmeans.labels_

plt.figure(figsize=(8, 5))
sns.scatterplot(x=merged_df['Registration Number'], y=merged_df['Average Marks'], hue=merged_df['Cluster'], palette='viridis')
plt.title('K-Means Clustering of Students Based on Average Marks')
plt.show()
