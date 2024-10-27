import pandas as pd
from io import StringIO

# Read and truncate the dataset to extract two batches of 5000 unique users
input_file = './sd/gowalla/original-Gowalla_totalCheckins.txt'  # Path to the original file
batch1_output_file = './sd/gowalla/Gowalla_totalCheckins.txt'  # Path to save the first 5000 users' data
batch2_output_file = './sd/gowalla/test-dataset.txt'  # Path to save the second 5000 users' data

# Load the dataset
with open(input_file, 'r') as infile:
    lines = infile.readlines()

# Assuming the format is tab-separated values
df = pd.read_csv(StringIO(''.join(lines)), sep='\t', header=None, names=["user_id", "date-time", "latitude", "longitude", "location_id"])

# Get the unique user IDs
unique_users = df['user_id'].unique()

# Ensure there are at least 10,000 unique users in the dataset
if len(unique_users) < 2720:
    raise ValueError("The dataset does not contain enough users to create two batches of 5000 users.")

# Get the first 5000 and second 5000 unique users
# batch1_users = unique_users[:23800]
# batch2_users = unique_users[23800:81400]
batch1_users = unique_users[:1000]
batch2_users = unique_users[1000:2720]

# Filter out the data for the two batches of users
batch1_users_data = df[df['user_id'].isin(batch1_users)]
batch2_users_data = df[df['user_id'].isin(batch2_users)]

# Save the first 5000 users' dataset to a new file
with open(batch1_output_file, 'w') as outfile:
    batch1_users_data.to_csv(outfile, sep='\t', header=False, index=False)

# Save the second 5000 users' dataset to a new file
with open(batch2_output_file, 'w') as outfile:
    batch2_users_data.to_csv(outfile, sep='\t', header=False, index=False)

print(f'Batch1: {len(batch1_users_data)}')
print(f'Batch1: {len(batch2_users_data)}')
