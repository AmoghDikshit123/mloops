import pandas as pd
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target column
df['target'] = iris.target

# Save the dataset to a CSV file
df.to_csv("iris_dataset.csv", index=False)

print("Dataset saved as iris_dataset.csv")