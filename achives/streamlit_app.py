import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
from crepe import CREPE

# Initialize CREPE model
crepe = CREPE(model_capacity="tiny")

# Streamlit App
st.title("Audio Pitch Analysis with CREPE")

# Upload audio file
uploaded_file = st.file_uploader(
    "Upload an audio file", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    # Load the audio file using librosa
    audio, sr = librosa.load(uploaded_file, sr=16000, mono=True)
    audio = torch.tensor(audio).unsqueeze(0)

    # Run CREPE pitch analysis
    time, frequency, confidence, activation = crepe.predict(audio, sr)

    # Spectrogram
    fig, ax = plt.subplots(figsize=(10, 6))

    # Display the spectrogram using linear frequency scale
    S = librosa.stft(y=audio[0, :].detach().numpy(), n_fft=512, hop_length=512)
    S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    img = librosa.display.specshow(
        S_dB, sr=sr, x_axis='time', y_axis='linear', ax=ax)

    # Overlay pitch
    ax.plot(time, frequency.detach().numpy(),
            label='Pitch (CREPE)', color='cyan', linewidth=2)

    # Add labels and color bar
    ax.set(title='Spectrogram with Pitch Overlay', ylabel='Frequency (Hz)')
    ax.label_outer()
    ax.legend(loc='upper right')

    st.pyplot(fig)

    # Optionally, save the plot
    save_plot = st.checkbox("Save the plot")
    if save_plot:
        plot_file = "spectrogram_with_pitch.png"
        fig.savefig(plot_file)
        st.success(f"Plot saved as {plot_file}")
