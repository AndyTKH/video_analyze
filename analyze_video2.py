import streamlit as st
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import librosa
import plotly.graph_objects as go
import moviepy.editor as mp
from fer import Video
from fer import FER
import matplotlib.pyplot as plt
import base64
import tempfile
import glob
import os
from pathlib import Path
from PIL import Image


# Define functions
def vidtoaudio(filename, output_name):
    video = mp.VideoFileClip(filename)
    video.audio.write_audiofile(output_name)


@st.cache_data(show_spinner=False)
def load_audio_features(mp3file):
    X, sample_rate = librosa.load(mp3file, sr=None)
    if X.ndim > 1:
        X = X[:, 0]
    X = X.T

    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20).T, axis=0)
    rmse = np.mean(librosa.feature.rms(y=X).T, axis=0)
    spectral_flux = np.mean(librosa.onset.onset_strength(y=X, sr=sample_rate).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)

    extracted_features = np.hstack([mfccs, rmse, spectral_flux, zcr])
    return extracted_features.reshape(1, -1)


@st.cache_data(show_spinner=False)
def predict_flu(mp3file, _svm_clf): 
    features = load_audio_features(mp3file)
    return _svm_clf.predict(features)


def visualize(l2, dd):
    fig = go.Figure(go.Indicator(
        mode="gauge",
        domain={'x': [0, 1], 'y': [0, 1]},
        title=dd,
        gauge={
            'axis': {'range': [None, 3], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': l2,
        }
    ))
    st.plotly_chart(fig)


def analyze_emotions(videofile):
    detector = FER(mtcnn=True)
    video = Video(videofile)
    raw_data = video.analyze(detector, display=False, save_fps=3)
    df = video.to_pandas(raw_data)
    df = video.get_first_face(df)
    df = video.get_emotions(df)

    per = []
    for i in df.columns:
        per.append(df[i].sum() / (df.sum().sum()) * 100)

    label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    plt.barh(label, per, color=['red', 'pink', 'orange', 'cyan', 'lime', 'yellow', 'gray'])
    plt.title('Emotion Chart')
    plt.ylabel('')
    plt.xlabel('Emotions %')
    st.pyplot(plt)
    plt.close()  # Release memory used by plt


def get_latest_file(folder, extension):
    list_of_files = glob.glob(os.path.join(folder, f'*.{extension}'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)


def reencode_video_moviepy(input_path, output_path):
    try:
        video = mp.VideoFileClip(input_path)
        video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    except Exception as e:
        st.error(f"An error occurred during re-encoding: {e}")


# Load pre-trained SVM model
feat_path = Path(__file__).parent / 'feat.npy'
label_path = Path(__file__).parent / 'label.npy'


@st.cache_resource(show_spinner=False)
def load_model_and_data():
    X = np.load(feat_path)
    y = np.load(label_path).ravel()
    np.random.seed(7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)
    svm_clf = SVC(C=200, gamma=0.0001, kernel='rbf', decision_function_shape="ovr")
    svm_clf.fit(X_train, y_train)
    return svm_clf


svm_clf = load_model_and_data()

# Streamlit app
st.title("Speech Analyzer")

# Add file uploader to sidebar
st.sidebar.header("Upload Video")
uploaded_file = st.sidebar.file_uploader("Upload an MP4 file", type=["mp4"])

# Create two columns with custom width ratios
col1, col2 = st.columns([2, 2])
col1.write("Analysis of Rishi Sunak's Speech:")
col2.write("Analysis of Your Speech:")

# Define the image paths
face_img_path = Path(__file__).parent / 'pic/face.png'
emo_img_path = Path(__file__).parent / 'pic/emo.png'
flu_img_path = Path(__file__).parent / 'pic/flu.png'

with col1:
    # Open and display face image
    with face_img_path.open('rb') as face_file:
        face_img = Image.open(face_file)
        st.image(face_img, use_column_width=True)
    
    # Open and display emotion image
    with emo_img_path.open('rb') as emo_file:
        emo_img = Image.open(emo_file)
        st.image(emo_img, use_column_width=True)

    # Open and display fluency image
    with flu_img_path.open('rb') as flu_file:
        flu_img = Image.open(flu_file)
        st.image(flu_img, use_column_width=True)


with col2:
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio_path = temp_audio.name

        # Extract audio from video
        vidtoaudio(temp_video_path, temp_audio_path)

        # Predict fluency
        y_predict = predict_flu(temp_audio_path, svm_clf)

        if y_predict == 0:
            visualize([{'range': [0, 1], 'color': 'red'}], {'text': "English Fluency on Beginner level", 'font': {'size': 15}})
        elif y_predict == 1:
            visualize([{'range': [0, 1], 'color': 'red'}, {'range': [1, 2], 'color': 'yellow'}], {'text': "English Fluency on Intermediate level", 'font': {'size': 15}})
        else:
            visualize([{'range': [0, 1], 'color': 'red'}, {'range': [1, 2], 'color': 'yellow'}, {'range': [2, 3], 'color': 'green'}], {'text': "English Fluency on Advanced level", 'font': {'size': 15}})

        # Analyze emotions
        analyze_emotions(temp_video_path)

        # Get the latest video file from the output folder
        latest_video_file = get_latest_file("output", "mp4")

        if latest_video_file:
            if os.path.exists(latest_video_file):
                # Re-encode the video
                reencoded_video_path = os.path.join("output", "reencoded_output.mp4")
                reencode_video_moviepy(latest_video_file, reencoded_video_path)

                # Display the re-encoded video
                if os.path.exists(reencoded_video_path):
                    st.video(reencoded_video_path)
                else:
                    st.error("Re-encoded video file does not exist.")
            else:
                st.error("Processed video file does not exist.")
        else:
            st.error("No processed video found.")
