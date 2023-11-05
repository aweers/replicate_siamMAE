import numpy as np
import cv2
import os

# Define the video settings
width, height = 300, 300
fps = 30
seconds = 10
output_path = "framenumbers/class1/"

def create_video(id=0):
    video_id = str(id)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(output_path + video_id + '.mp4', fourcc, fps, (width, height))

    for frame_number in range(fps * seconds):
        # Create a black image and add the frame numbers
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw the frame number on the image
        cv2.putText(img, video_id + ": " + str(frame_number), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Write the frame to the video
        video.write(img)

    video.release()

if not os.path.exists(output_path):
    os.makedirs(output_path)

for i in range(20):
    create_video(i)