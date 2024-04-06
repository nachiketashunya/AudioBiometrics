from sklearn.mixture import GaussianMixture
import joblib
import os
import numpy as np
from .preprocess import AudioPreprocessor

ENROLL_DIR = "enroll/"
enroll_samples = os.lisdir(ENROLL_DIR)

# Enrollment loop
preprocessed_features = []
for i in range(len(enroll_samples)):
    # Assume user provides the audio sample and it is stored in audio_samples[i]
    audio_sample = enroll_samples[i]  # You need to provide the actual audio sample

    # Preprocess the audio sample
    processor = AudioPreprocessor.preprocess_audio()
    processed_audio = processor.preprocess_audio(audio_sample)

    # Append preprocessed features to the list
    preprocessed_features.extend(processed_audio)

# Convert preprocessed features to a numpy array
preprocessed_features_array = np.array(preprocessed_features)

# Train Gaussian Mixture Model (GMM)
num_components = 5  # Example number of components
gmm = GaussianMixture(n_components=num_components)
gmm.fit(preprocessed_features_array)

# Save the trained GMM model
joblib.dump(gmm, 'trained_gmm_model.pkl')