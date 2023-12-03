#!/bin/bash

# Source and target directories
src_dir="data_20/class1"
target_dir="frames_20/class1"

# Ensure target directory exists
mkdir -p "$target_dir"

# Count files
num_files=$(ls -1 "$src_dir"/*.mp4 | wc -l)

# Start timer
start=$(date +%s)

# Iterate over each video file in the source directory
counter=0
for file in "$src_dir"/*.mp4; do
    #if [ $counter -ge 100 ]; then
    #    break
    #fi
    # Extract the base name of the file (without extension)
    base_name=$(basename "$file" .mp4)

    # Create a new subdirectory in the target directory for this video
    mkdir -p "$target_dir/$base_name"

    # Use ffmpeg to extract and resize the frames
    #ffmpeg -i "$file" -loglevel error -vf "scale=-1:224" "$target_dir/$base_name/frame%04d.png" &               # 39s/100  3.9GB/100
    #ffmpeg -i "$file" -loglevel error -vf "fps=15,scale=-1:224" -q:v 1 "$target_dir/$base_name/frame%04d.jpg" &  # 13s/100  250MB/100
    (ffmpeg -i "$file" -loglevel error -vf "fps=15,scale=-1:224" -q:v 1 "$target_dir/$base_name/frame%04d.jpg" && rm "$file") &
    
    while (( $(jobs -p | wc -l) >= 8 )); do
        sleep 0.1
    done
    ((counter++))

    echo -ne "Processed $counter files out of $num_files\r"
done

wait

end=$(date +%s)
elapsed=$((end - start))
echo "Extracted frames of $num_files videos in $elapsed seconds."