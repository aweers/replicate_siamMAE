#!/bin/bash

# Default value
delete_files="n"

# Process command line arguments
while (( "$#" )); do
  case "$1" in
    -y)
      echo "Warning: This script may delete files!"
      delete_files="y"
      shift
      ;;
    *)
      echo "Invalid argument: $1"
      exit 1
      ;;
  esac
done

# 29-30 fps and minimum 3s
count_group1=0
# 23.9-29 fps and minimum 3s
count_group2=0
# all
count_total=0
non_group1_files=()

# Loop over all .mp4 files in the current directory
for file in *.mp4; do
  # get fps
  fps=$(ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate "./$file" 2>/dev/null | bc)
  if [ $? -ne 0 ]; then
    echo "Corrupt file: ./$file"
    non_group1_files+=("$file")
    continue
  fi

  # get duration
  duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "./$file" 2>/dev/null)
  if [ $? -ne 0 ]; then
    echo "Corrupt file: ./$file"
    non_group1_files+=("$file")
    continue
  fi

  # Check if the fps is within the desired range and the duration is as desired
  if (( $(echo "$duration >= 3" | bc -l) )); then
    if (( $(echo "$fps >= 29 && $fps <= 30" | bc -l) )); then
      ((count_group1++))
    elif (( $(echo "$fps >= 23.9 && $fps < 29" | bc -l) )); then
      ((count_group2++))
      non_group1_files+=("$file")
    else
      non_group1_files+=("$file")
    fi
  else
    non_group1_files+=("$file")
  fi

  ((count_total++))

  # Print the results continuously
  echo -ne "Group 1 count: $count_group1, Group 2 count: $count_group2, Non-Group 1 count: ${#non_group1_files[@]}, Total count: $count_total\r"
done

# Print the final results
echo -e "\nFinal counts:"
echo "Number of files with minimum 3s and fps between 29 and 30: $count_group1"
echo "Number of files with minimum 3s and fps between 23.9 and 29: $count_group2"
echo "Number of non-Group 1 files: ${#non_group1_files[@]}"
echo "Total number of files: $count_total"

# Check whether to delete files not in group 1
if [[ "$delete_files" == "y" ]]; then
  echo "Deleting files..."
else
  # Ask the user
  read -p "Delete all files not in Group 1? [N/y] " delete_files
  if [[ "$delete_files" =~ ^[Yy]$ ]]; then
    echo "Deleting files..."
  else
    echo "Not deleting files."
    exit 0
  fi
fi

# Delete the files
for file in "${non_group1_files[@]}"; do
  rm "./$file"
done
echo "Files deleted."