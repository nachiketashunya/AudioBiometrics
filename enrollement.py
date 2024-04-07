from sklearn.mixture import GaussianMixture
import joblib
import os
import numpy as np
from preprocess import AudioPreprocessor
from dataset import get_enroll_ds, extract_features
import pandas as pd

ENROLL_DIR = "enroll/"
DATASET_DIR = "data/svarah/audio"
enroll_samples = os.lisdir(ENROLL_DIR)

enroll_id, enroll_df = get_enroll_ds()

audio_df = pd.DataFrame()
for index, row in enroll_df.iterrows():
  audio_file = os.path.join("/content/svarah/svarah", row['audio_filepath'])

  features = extract_features(enroll_id, audio_file)
  audio_df = pd.concat([audio_df, features], ignore_index=True)


enroll_features = audio_df.drop(columns=['filename', 'speaker_id'])
# Train Gaussian Mixture Model (GMM)
num_components = 2 # Example number of components
gmm = GaussianMixture(n_components=num_components)
gmm.fit(enroll_features)

# Save the trained GMM model
joblib.dump(gmm, f'models/{enroll_id}_gmm_model.pkl')