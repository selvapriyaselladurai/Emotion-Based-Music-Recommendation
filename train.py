import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import joblib

train_dir = "emotion-dataset-demo/train"
test_dir = "emotion-dataset-demo/test"

IMG_SIZE = 48

emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def load_data(data_dir):
    images = []
    labels = []
    for label in emotion_labels:
        emotion_path = os.path.join(data_dir, label)
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img_resized.flatten())  # Flatten for SVM input
                labels.append(label)
    return np.array(images), np.array(labels)

X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(kernel='linear', probability=True)
model.fit(X_train_scaled, y_train_encoded)

joblib.dump(model, 'emotion_svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Training complete. Model saved as 'emotion_svm_model.pkl'.")
