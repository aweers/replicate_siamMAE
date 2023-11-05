#!/bin/bash

# Process arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <filename> <number of files> <directory>"
    exit 1
fi
filename=$1
num_files=$2
directory=$3

# Create the directory if it does not exist
mkdir -p $directory

# Calculate the step size for uniform sampling
total_lines=$(wc -l < $filename)
step_size=$((total_lines / num_files))

# Start timer
start=$(date +%s)

counter=1
file_counter=0
# Read the file line by line
while IFS= read -r line
do
    # Download and extract every "step_size"th file
    if (( counter % step_size == 0 )); then
        progress=$(echo "scale=2; $file_counter * 100 / $num_files" | bc)
        echo "Downloading and extracting file from URL: $line ($progress% done)"
        wget -q --show-progress -O - "$line" | tar -xz -C $directory
    fi
    ((counter++))
done < $filename

end=$(date +%s)
elapsed=$((end - start))
echo "Downloaded and extracted files in $elapsed seconds."