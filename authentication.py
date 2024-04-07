import numpy as np
import os
from .dataset import extract_features
import torchaudio

SPEAKER_MODELS = os.listdir("models/")

auth_audio_file = "path"

auth_audio, _ = torchaudio.load(auth_audio_file)
auth_ext_feat = extract_features(auth_audio)

log_likelihood_speakers = np.zeros(len(SPEAKER_MODELS))

for i in range(len(SPEAKER_MODELS)):
  gmm = SPEAKER_MODELS[i]  # checking with each model one by one
  scores = np.array(gmm.score(auth_ext_feat))  # get the score
  log_likelihood_speakers[i] = scores.sum()  # add model score to the array

 # others detection with threshold
normalized_possibility = max(log_likelihood_speakers) - log_likelihood_speakers
not_others_flag = True

for k in range(len(normalized_possibility)):
    if log_likelihood_speakers[k] == max(log_likelihood_speakers):
        continue

    if abs(normalized_possibility[k]) < 0.28:
        not_others_flag = False

if not_others_flag:
    prediction_speaker = np.argmax(log_likelihood_speakers)

else:
    prediction_speaker = -1

print(prediction_speaker)