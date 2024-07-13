import pandas as pd

# Path to your .txt dataset
dataset_path = "/Users/kalpithanaik/Desktop/medbot/att3.py/training_data/train.txt"

# Read the dataset
with open(dataset_path, 'r') as file:
    lines = file.readlines()

# Split each line into symptoms and condition
data = []
for line in lines:
    parts = line.strip().split(',')
    if len(parts) == 2:  # Ensure there are exactly two parts
        data.append(parts)

# Create a DataFrame
df = pd.DataFrame(data, columns=["symptoms", "condition"])

# Save the DataFrame to a CSV file
csv_path = "/Users/kalpithanaik/Desktop/medbot/att3.py/training_data/train.csv"
df.to_csv(csv_path, index=False)

print(f"Dataset saved to {csv_path}")
