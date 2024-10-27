import pandas as pd
from io import StringIO

users = '1000'

# Read and truncate the dataset to exclude the first 100 unique users
input_file = './small-dataset/gowalla/original-Gowalla_totalCheckins.txt'  # Path to the original file
firstBatch = './small-dataset/gowalla/fb-Gowalla_totalCheckins.txt'  # Path to save the remaining user data
secondBatch = './small-dataset/gowalla/sb-Gowalla_totalCheckins.txt'  # Path to save the remaining user data

# Load the dataset
with open(input_file, 'r') as infile:
    lines = infile.readlines()

# Assuming the format is tab-separated values
df = pd.read_csv(StringIO(''.join(lines)), sep='\t', header=None, names=["user_id", "date-time", "latitude", "longitude", "location_id"])

# Get the unique user IDs
unique_users = df['user_id'].unique()

# Separate the first 100 users
fb = unique_users[:users]
sb = unique_users[users:2000]

# Filter out the excluded users
fb_data = df[df['user_id'].isin(fb)]
sb_data = df[df['user_id'].isin(fb)]

# Save the excluded users' dataset to a new file
with open(firstBatch, 'w') as outfile:
    fb_data.to_csv(outfile, sep='\t', header=False, index=False)

with open(secondBatch, 'w') as outfile:
    fb_data.to_csv(outfile, sep='\t', header=False, index=False)

