import os
import shutil
import random

# Set the path to the input folder
input_folder = 'input'

# Create the subfolders
train_folder = os.path.join(input_folder, 'train')
test_folder = os.path.join(input_folder, 'test')
val_folder = os.path.join(input_folder, 'val')

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Create subfolders for each class within the train, test, and val folders
train_risky_folder = os.path.join(train_folder, 'risky')
train_nonrisky_folder = os.path.join(train_folder, 'nonrisky')
test_risky_folder = os.path.join(test_folder, 'risky')
test_nonrisky_folder = os.path.join(test_folder, 'nonrisky')
val_risky_folder = os.path.join(val_folder, 'risky')
val_nonrisky_folder = os.path.join(val_folder, 'nonrisky')

os.makedirs(train_risky_folder, exist_ok=True)
os.makedirs(train_nonrisky_folder, exist_ok=True)
os.makedirs(test_risky_folder, exist_ok=True)
os.makedirs(test_nonrisky_folder, exist_ok=True)
os.makedirs(val_risky_folder, exist_ok=True)
os.makedirs(val_nonrisky_folder, exist_ok=True)

# Get the list of files in the input folder
files = os.listdir(input_folder)

# Filter out non-AVI files
avi_files = [file for file in files if file.endswith('.avi')]

# Separate files by label and activity
risky_files = {}
nonrisky_files = {}

for file in avi_files:
    label = file.split('_')[-1][:-4]  # Extract the label from the filename
    activity = file.split('_')[0].split('A')[1]  # Extract the activity number from the filename
    
    if label == 'risky':
        if activity not in risky_files:
            risky_files[activity] = []
        risky_files[activity].append(file)
    elif label == 'nonrisky':
        if activity not in nonrisky_files:
            nonrisky_files[activity] = []
        nonrisky_files[activity].append(file)

# Function to divide files into train, test, and validation sets
def divide_files(files):
    train_files = []
    val_files = []
    test_files = []
    
    for activity, activity_files in files.items():
        random.shuffle(activity_files)
        num_files = len(activity_files)
        train_split = int(0.6 * num_files)
        val_split = int(0.2 * num_files)
        
        train_files.extend(activity_files[:train_split])
        val_files.extend(activity_files[train_split:train_split+val_split])
        test_files.extend(activity_files[train_split+val_split:])
    
    return train_files, val_files, test_files

# Divide risky and nonrisky files into train, test, and validation sets
risky_train, risky_val, risky_test = divide_files(risky_files)
nonrisky_train, nonrisky_val, nonrisky_test = divide_files(nonrisky_files)

# Function to move files to the appropriate subfolder
def move_files(files, dest_folder):
    for file in files:
        shutil.move(os.path.join(input_folder, file), dest_folder)

# Move files to the respective subfolders
move_files(risky_train, train_risky_folder)
move_files(risky_val, val_risky_folder)
move_files(risky_test, test_risky_folder)
move_files(nonrisky_train, train_nonrisky_folder)
move_files(nonrisky_val, val_nonrisky_folder)
move_files(nonrisky_test, test_nonrisky_folder)

print("Files have been successfully divided into train, test, and validation sets with balanced labels and activities.")