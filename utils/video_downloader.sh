#!/bin/bash

# Process arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <filename> <number of files> <directory>"
    exit 1
fi
filename=$1
num_files=$2
directory="${3}/class1"
val_directory="${3}_val/class1"

# Create the directories if they do not exist
mkdir -p $directory
mkdir -p $val_directory

# Calculate the step size for uniform sampling
total_lines=$(wc -l < $filename)
step_size=$((total_lines / num_files))

# Calculate the number of files for each directory
num_files_train=$((num_files * 7 / 10))
num_files_val=$((num_files - num_files_train))

echo "Downloading and extracting $num_files_train files to $directory and $num_files_val files to $val_directory."

# Start timer
start=$(date +%s)

counter=1
file_counter=0
val_file_counter=0
# Read the file line by line
while IFS= read -r line
do
    # Download and extract every "step_size"th file
    if (( counter % step_size == 0 )); then
        if (( file_counter < num_files_train )); then
            download_directory=$directory
            ((file_counter++))
        else
            download_directory=$val_directory
            ((val_file_counter++))
        fi
        progress=$(echo "scale=2; ($file_counter + $val_file_counter) * 100 / $num_files" | bc)
        echo "Downloading and extracting file from URL: $line ($progress% done)"
        wget -q --show-progress -O - "$line" | tar -xz -C $download_directory
    fi
    ((counter++))
done < $filename

end=$(date +%s)
elapsed=$((end - start))
echo "Downloaded and extracted files in $elapsed seconds."
