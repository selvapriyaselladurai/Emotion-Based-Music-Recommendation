import cv2
import joblib
import numpy as np
import time
import threading
import pygame
import random
import os

# Load ML models and encoders
model = joblib.load('emotion_svm_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Emotion-to-music folder mapping
emotion_music_map = {
    "angry": "music/angry/",
    "disgust": "music/disgust/",
    "fear": "music/fear/",
    "happy": "music/happy/",
    "neutral": "music/neutral/",
    "sad": "music/sad/",
    "surprise": "music/surprise/"
}

# Emotion colors for UI
emotion_colors = {
    "angry": (0, 0, 255),
    "disgust": (128, 0, 128),
    "fear": (0, 255, 255),
    "happy": (0, 255, 0),
    "neutral": (192, 192, 192),
    "sad": (255, 0, 0),
    "surprise": (255, 255, 0)
}

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize pygame mixer
pygame.mixer.init()

is_playing_music = False
video_paused = False

# Track last played song per emotion
last_played_song = {emotion: None for emotion in emotion_music_map}

def get_random_song(emotion):
    """Returns a random song from the emotion folder, avoiding the last played song."""
    folder = emotion_music_map.get(emotion)  # Ensure valid key lookup

    if not folder or not os.path.exists(folder):
        print(f"Error: Folder not found for emotion '{emotion}' at {folder}")
        return None

    songs = [f for f in os.listdir(folder) if f.endswith('.mp3')]

    if not songs:
        print(f"Error: No songs found for emotion '{emotion}'")
        return None

    new_song = random.choice(songs)
    
    # Ensure new song is different from the last played song (if multiple songs exist)
    while new_song == last_played_song.get(emotion) and len(songs) > 1:
        new_song = random.choice(songs)

    last_played_song[emotion] = new_song
    return os.path.join(folder, new_song)

def play_music(emotion):
    """Plays a random song for the detected emotion."""
    global is_playing_music
    song = get_random_song(emotion)

    if not song:  # Check if a valid song was returned
        print(f"No valid song available for emotion: {emotion}")
        return  

    try:
        pygame.mixer.music.load(song)
        pygame.mixer.music.play()
        is_playing_music = True
        print(f"Playing: {song}")
    except Exception as e:
        print(f"Error playing {song}: {e}")

def stop_music_after_delay():
    """Stops music after 20 seconds and resumes video detection."""
    global is_playing_music
    time.sleep(20)
    pygame.mixer.music.stop()
    is_playing_music = False
    print("Music stopped.")
    resume_video_detection()

def start_music_timer():
    """Starts a thread to stop music after a delay."""
    timer_thread = threading.Thread(target=stop_music_after_delay)
    timer_thread.start()

def pause_video_detection():
    """Pauses face detection while music plays."""
    global video_paused
    video_paused = True

def resume_video_detection():
    """Resumes face detection after music stops."""
    global video_paused
    video_paused = False

# Initialize emotion count
emotion_count = {emotion: 0 for emotion in emotion_colors}
total_detections = 0

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    if video_paused:
        time.sleep(1)
        continue

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48)).flatten().reshape(1, -1)
        face_scaled = scaler.transform(face_resized)

        emotion_prediction = model.predict(face_scaled)
        emotion_label = label_encoder.inverse_transform(emotion_prediction)[0]

        if not is_playing_music:
            play_music(emotion_label)
            start_music_timer()
            pause_video_detection()

        color = emotion_colors.get(emotion_label, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{emotion_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Emotion Detection with Music Playback', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quit requested. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
