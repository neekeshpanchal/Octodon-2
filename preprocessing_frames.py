import cv2
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Create a function to process the video
def process_video():
    # Ask the user to select a video file
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if not file_path:
        return

    try:
        # Load the video
        video_capture = cv2.VideoCapture(file_path)

        # Create an empty DataFrame to store sector data
        num_sectors = 64 * 64
        columns = [f'Sector_{i+1}' for i in range(num_sectors)]
        data = []

        # Read and process each frame
        frame_number = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break  # Break the loop at the end of the video

            # Split the frame into sectors
            height, width, _ = frame.shape
            sector_height = height // 64
            sector_width = width // 64

            sector_data = []
            for i in range(64):
                for j in range(64):
                    sector = frame[i * sector_height:(i + 1) * sector_height, j * sector_width:(j + 1) * sector_width]
                    sector_avg_rgb = tuple(np.mean(sector, axis=(0, 1)))[::-1]  # Calculate the average RGB value and reverse the order to (B, G, R)

                    # Check if the RGB value is within the specified range
                    if (64, 72, 64) <= sector_avg_rgb <= (96, 96, 96):
                        sector_data.append(sector_avg_rgb)  # Highlighted cells are added to the list
                    else:
                        sector_data.append('')  # Empty cell if not in range

            data.append(sector_data)

            frame_number += 1

        # Create a DataFrame and store it in a CSV file
        df = pd.DataFrame(data, columns=columns)
        output_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if output_file:
            df.to_csv(output_file, index=False)  # Save the data to a CSV file

        # Release the video capture object
        video_capture.release()

        messagebox.showinfo("Processing Complete", f'Processed {frame_number} frames. Data saved to {output_file}')
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main application window
root = tk.Tk()
root.title("Video Processing")

# Create a button to browse and process the video
process_button = tk.Button(root, text="Process Video", command=process_video)
process_button.pack(padx=10, pady=10)

# Start the main event loop
root.mainloop()
