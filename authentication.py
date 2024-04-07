import numpy as np
import os
from dataset import extract_features
import joblib
from dataset import get_mapping

SPEAKER_MODELS = os.listdir("models/")
DATASET_DIR = "/data/svarah/audio"

def map_estimation(gmm_models, features):
    log_likelihood_speakers = np.zeros(len(gmm_models))
    model_names = []

    # Calculate log likelihoods for each speaker model
    for i, gmm in enumerate(gmm_models):

        if gmm.endswith(".pkl"):
          model_names.append(gmm)
          gmm = joblib.load(os.path.join("/content/models", gmm))
          scores = gmm.score(features)
          # print(scores)
          log_likelihood_speakers[i] = scores.sum()

    # Compute prior probabilities (assuming uniform priors)
    prior_probabilities = np.ones(len(gmm_models)) / len(gmm_models)

    # Compute posterior probabilities using Bayes' theorem
    log_posterior_probabilities = log_likelihood_speakers + np.log(prior_probabilities)

    # Get the predicted speaker ID with the maximum posterior probability
    predicted_index = np.argmax(log_posterior_probabilities)
    predicted_speaker = model_names[predicted_index].split("_")[0]

    return predicted_speaker


def predidc(file_name):
  auth_audio_file = os.path.join(DATASET_DIR, file_name)
  # auth_audio, _ = torchaudio.load(auth_audio_file)
  auth_ext_feat = extract_features(file=auth_audio_file)
  auth_ext_feat = auth_ext_feat.drop(columns=['filename', 'speaker_id'])
  prediction_speaker = map_estimation(speakers_models, auth_ext_feat)

  map = file_name.split("_")[0]
  mapping = get_mapping()

  return [prediction_speaker, mapping.get(map)]

speakers_models = os.listdir(SPEAKER_MODELS)

file_name = "281474976883943_f2231_chunk_2.wav"
result = predidc(file_name)
print(f"Predicted Speaker ID: {result[0]}\nOriginal Speaker ID: {result[1]}")

file_name = "281474976888866_f2195_chunk_16.wav"
result = predidc(file_name)
print(f"Predicted Speaker ID: {result[0]}\nOriginal Speaker ID: {result[1]}")
