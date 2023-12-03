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

num_files=$(ls -1 . | wc -l)
counter=0
# Loop over all .mp4 files in the current directory
for file in *.mp4; do
  (
    # get fps
    fps=$(ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate "./$file" 2>/dev/null | bc)
    if [ $? -ne 0 ]; then
      echo "Corrupt file: ./$file"
      rm "./$file"
      continue
    fi

    # get duration
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "./$file" 2>/dev/null)
    if [ $? -ne 0 ]; then
      echo "Corrupt file: ./$file"
      rm "./$file"
      continue
    fi

    # Check if the fps is within the desired range and the duration is as desired
    if (( $(echo "$duration >= 3" | bc -l) )); then
      if (( $(echo "$fps >= 29 && $fps <= 30" | bc -l) )); then
        echo "Keeping $file"
      else
        echo "Deleting $file"
        rm "./$file"
      fi
    else
      echo "Deleting $file"
      rm "./$file"
    fi
  ) &

  # Limit the number of concurrent jobs
  while (( $(jobs -p | wc -l) >= 8 )); do
    sleep 0.1
  done
  ((counter++))

  echo -ne "Processed $counter files out of $num_files\r"
done

# Wait for all background jobs to finish
wait

echo "Processing complete."