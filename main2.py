import base64
import os
import requests
import json
import spotipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time
import numpy as np
from datetime import datetime

# Spotipy setup
ourscope = "user-read-currently-playing user-modify-playback-state user-read-playback-state playlist-read-private"
mgr = spotipy.oauth2.SpotifyOAuth(client_id="c2ff0e35755e4337a7ac709d713d6a91", client_secret="a2520a28dfe34835b4396a5c87ce7a9c", redirect_uri="https://google.com/", scope=ourscope)
sp = spotipy.Spotify(auth_manager=mgr)

# Define the VisionRecognitionModel architecture
class VisionRecognitionModel(nn.Module):
    def __init__(self):
        super(VisionRecognitionModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 20)  # Adjust the number of classes as needed
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Define the GestureRecognitionModel architecture
class GestureRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GestureRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=5, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.bn(lstm_out)
        fc1_out = self.fc1(lstm_out)
        fc1_out = F.relu(fc1_out)
        fc2_out = self.fc2(fc1_out)
        fc2_out = F.relu(fc2_out)
        output = self.fc3(fc2_out)
        return output

# Load the visual gesture recognition model
visual_model = VisionRecognitionModel()
visual_model.load_state_dict(torch.load('visual_gesture_recognition_model_fold1.pth'))
visual_model.eval()

# Load the distance sensor gesture recognition model
input_size = 3  # Update this based on your input size
hidden_size = 100
output_size = 3  # Update this based on your number of classes

distance_model = GestureRecognitionModel(input_size, hidden_size, output_size)
distance_model.load_state_dict(torch.load('gesture_recognition_model.pth'))
distance_model.eval()

# Function to read distance data from the log file
def read_distance_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if lines:
            last_line = lines[-1].strip()
            timestamp, distance = last_line.split(',')
            timestamp = float(timestamp)
            distance = float(distance)
            return timestamp, distance
    return None, None

# Function to preprocess the webcam frame for the visual model
def preprocess_frame(frame):
    # Convert the frame to RGB (if necessary)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize to 224x224, which the model expects
    resized = cv2.resize(rgb_frame, (224, 224))
    # Normalize the pixel values
    normalized = resized / 255.0
    # Convert to a tensor and add batch dimension
    tensor = torch.tensor(normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return tensor

# Function to detect visual gesture
def detect_visual_gesture(frame):
    preprocessed_frame = preprocess_frame(frame)
    with torch.no_grad():
        gesture = visual_model(preprocessed_frame)
    _, predicted_gesture = torch.max(gesture, 1)
    return predicted_gesture.item()

# Function to process the distance data with the distance model
def detect_distance_gesture(distance, delta_distance, time_diff):
    features = torch.tensor([[distance, delta_distance, time_diff]], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        gesture = distance_model(features)
    _, predicted_gesture = torch.max(gesture, 1)
    return predicted_gesture.item()

# Function to control Spotify based on gestures
def control_spotify(visual_gesture, distance_gesture):
    devices = sp.devices()
    if not devices['devices']:
        print("No active device found")
        return

    # Distance Gesture 0: Pause
    if distance_gesture == 0:
        current_playback = sp.current_playback()
        if current_playback and current_playback['is_playing']:
            sp.pause_playback()
            print("Pause")
        else:
            sp.start_playback()
            print("Play")

    # Visual Gestures
    if visual_gesture == 0:  # Play
        sp.start_playback()
        print("Play")
    elif visual_gesture == 1:  # Forward 15 seconds
        current_playback = sp.current_playback()
        if current_playback and current_playback['is_playing']:
            current_position = current_playback['progress_ms']
            new_position = min(current_position + 15000, current_playback['item']['duration_ms'])
            sp.seek_track(new_position)
            print("Forward 15 seconds")
    elif visual_gesture == 2:  # Rewind 15 seconds
        current_playback = sp.current_playback()
        if current_playback and current_playback['is_playing']:
            current_position = current_playback['progress_ms']
            new_position = max(current_position - 15000, 0)
            sp.seek_track(new_position)
            print("Rewind 15 seconds")
    elif visual_gesture == 3:  # Next track
        sp.next_track()
        print("Next Track")
    elif visual_gesture == 4:  # Previous track
        sp.previous_track()
        print("Previous Track")
    elif visual_gesture == 5:  # Replay track
        sp.seek_track(0)
        print("Replay Track")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables to store previous distance and timestamp
previous_distance = None
previous_timestamp = None

# Main loop
while True:
    # Read the latest distance from the log file
    timestamp, distance = read_distance_log('C:\\Users\\gordo\\AppData\\Local\\teraterm5\\teraterm.log')
    if distance is None:
        continue

    # Calculate delta_distance and time_diff
    if previous_distance is not None and previous_timestamp is not None:
        delta_distance = distance - previous_distance
        time_diff = (timestamp - previous_timestamp) / 1000.0  # Convert to seconds
    else:
        delta_distance = 0.0
        time_diff = 0.0

    # Update previous distance and timestamp
    previous_distance = distance
    previous_timestamp = timestamp

    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Detect visual gesture
    visual_gesture = detect_visual_gesture(frame)

    # Detect distance gesture
    distance_gesture = detect_distance_gesture(distance, delta_distance, time_diff)

    # Combine the gesture information and control Spotify
    control_spotify(visual_gesture, distance_gesture)

    # Display the frame (optional)
    cv2.imshow('Webcam', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Add a small delay to avoid excessive CPU usage
    time.sleep(0.1)

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
