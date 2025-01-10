import tensorflow as tf
import tensorflow_datasets as tfds
import imageio
import os
import csv
import numpy as np

# output_dir = "droid_videos"
output_dir = '/data/droid/droid_videos'
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "video_labels.csv")
# dataset_name = "droid_100"
dataset_name = 'droid'
dataset_dir = "/home/madhavan/latent_actions/data/droid"  
# dataset_dir = '/data/droid/'
dataset = tfds.load(dataset_name, data_dir=dataset_dir, split="train") 

# Open CSV file to log video paths and labels
with open(csv_path, mode='a', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=' ')
    print(len(dataset))
    exit()
    for idx, episode in enumerate(dataset):
        # Define video file path and class label
        video_path = os.path.abspath(os.path.join(output_dir, f"video_{idx}.mp4"))

        steps = list(episode['steps'])
        frames = []
        class_labels = [steps[0]["language_instruction"].numpy().decode('utf-8')]
        class_labels.append(steps[0]["language_instruction_2"].numpy().decode('utf-8'))
        class_labels.append(steps[0]["language_instruction_3"].numpy().decode('utf-8'))
        class_label = None
        for c in class_labels:
            if c:
                class_label = c
                break
        if not class_label:
            class_label = str(idx)

        for step in steps:
            exterior_image_1 = step["observation"]["exterior_image_1_left"]
            # Resize to 224x224
            exterior_image_1 = tf.image.resize(exterior_image_1, (224, 224))
            # Ensure it's uint8 for saving to video
            # exterior_image_1 = tf.image.convert_image_dtype(exterior_image_1, dtype=tf.uint8)
            frames.append(exterior_image_1.numpy())

        # Save frames as a video
        with imageio.get_writer(video_path, fps=30, codec="libx264") as writer_video:
            for frame in frames:
                writer_video.append_data(frame.astype('uint8'))

        # Write the video path and label to CSV
        writer.writerow([f"{video_path}", class_label])
        print(f"Saved video and logged entry: {video_path}, {class_label}")

print("All videos have been saved and logged in CSV.")
