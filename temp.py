import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
from skimage import morphology
from skimage.segmentation import watershed
from scipy import ndimage

frame_frame = None
fig = None
canvas = None
video_path = ""
cap = None
frame_count = 0
max_frame_count = 0
is_processing_video = False
is_paused = False
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Define the U-Net model
def create_unet_model(input_shape):
    model = models.Sequential()
    
    # Encoder
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Bottleneck
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))

    # Decoder
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(1, (1, 1), activation='sigmoid', padding='valid'))

    model.summary() 

    return model

def process_video_detection():
    global is_processing_video, cap, frame_count, max_frame_count, is_paused
    is_processing_video = True
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    max_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    # Define U-Net model
    unet_model = create_unet_model((256, 256, 3))

    while is_processing_video:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                is_processing_video = False
                break

            # Resize the frame to fit the U-Net model input shape
            resized_frame = cv2.resize(frame, (256, 256))

            # Predict segmentation mask using U-Net
            mask = unet_model.predict(np.expand_dims(resized_frame, axis=0))[0, :, :, 0]

            # Resize the mask to match the original frame size
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            # Normalize the mask to values between 0 and 1
            mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))

            # Apply the mask to the original frame
            segmented_frame = frame.copy()
            segmented_frame[:, :, 2] = (mask * 255).astype(np.uint8)  # Highlight detected regions in red channel

            # Draw bounding boxes based on the segmentation mask
            contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(segmented_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            display_frame(segmented_frame)
            frame_count += 1
            update_frame_info(frame_count)


def display_frame(frame):
    global canvas, fig
    fig.clf()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax1 = fig.add_subplot(111)
    ax1.imshow(frame)
    ax1.set_title("Detection and Tracking")
    ax1.axis("off")
    canvas.draw()

def open_file():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if video_path:
        start_processing_button["state"] = "normal"
        stop_processing_button["state"] = "normal"
        frame_count_label.config(text=f"Frame: 0")
        update_frame_info(0)

def start_processing():
    global is_processing_video, is_paused
    is_processing_video = True
    is_paused = False
    Thread(target=process_video_detection).start()  # Use process_video_detection for HOG + Linear SVM


def stop_processing():
    global is_processing_video
    is_processing_video = False

def pause_play():
    global is_paused
    is_paused = not is_paused

def update_frame_info(frame_number):
    frame_count_label.config(text=f"Frame: {frame_number} / {max_frame_count}")

def scroll_frame_forward(event):
    global frame_count
    if frame_count < max_frame_count:
        frame_count += 1
        update_frame_info(frame_count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        edge_detected_frame = detect_edges(frame)
        display_frames(frame, edge_detected_frame)

def scroll_frame_backward(event):
    global frame_count
    if frame_count > 0:
        frame_count -= 1
        update_frame_info(frame_count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        edge_detected_frame = detect_edges(frame)
        display_frames(frame, edge_detected_frame)

def go_to_frame():
    global frame_count
    frame_number = int(frame_entry.get())
    if 0 <= frame_number <= max_frame_count:
        frame_count = frame_number
        update_frame_info(frame_count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        edge_detected_frame = detect_edges(frame)
        display_frames(frame, edge_detected_frame)
    else:
        frame_entry.delete(0, tk.END)
        frame_entry.insert(0, str(frame_count))

def detect_edges(frame):
    # Add edge detection or other processing as needed
    # For now, just return the original frame
    return frame

def display_frames(frame, edge_detected_frame):
    # Add any additional processing or display logic as needed
    # For now, just display the original frame
    display_frame(frame)

def main():
    global is_processing_video, start_processing_button, stop_processing_button, fig, canvas, frame_count_label, frame_entry
    is_processing_video = False

    root = tk.Tk()
    root.title("Octodon")
    root.configure(bg="#800080")  
    open_video_button = tk.Button(root, text="Open Video", command=open_file)
    open_video_button.pack(pady=10)

    start_processing_button = tk.Button(root, text="Start Processing", command=start_processing)
    start_processing_button.pack()
    start_processing_button["state"] = "disabled"

    stop_processing_button = tk.Button(root, text="Stop Processing", command=stop_processing)
    stop_processing_button.pack()
    stop_processing_button["state"] = "disabled"

    pause_play_button = tk.Button(root, text="Pause/Play", command=pause_play)
    pause_play_button.pack()

    frame_frame = tk.Frame(root)
    frame_frame.pack(fill=tk.BOTH, expand=True)

    fig = plt.Figure(figsize=(10, 4), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=frame_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    frame_count_label = tk.Label(root, text="Frame: 0")
    frame_count_label.pack()

    root.bind("<Right>", scroll_frame_forward)
    root.bind("<Left>", scroll_frame_backward)

    frame_entry_label = tk.Label(root, text="Go to Frame:")
    frame_entry_label.pack()
    frame_entry = tk.Entry(root)
    frame_entry.pack()
    go_to_frame_button = tk.Button(root, text="Go to Frame", command=go_to_frame)
    go_to_frame_button.pack()

    root.geometry("1280x480")

    root.mainloop()

if __name__ == "__main__":
    main()
