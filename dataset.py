import os
import pandas as pd
import torchaudio
from .preprocess import AudioPreprocessor  # Import the AudioPreprocessor class from your preprocessing module
from .get_df import get_dataframe

DATA_DIR = "data/svarah/audio"

def extract_features(self, preprocessor, file):
    # Load the audio file
    waveform, _ = torchaudio.load(file)

    # Preprocess audio using the provided preprocessor
    ps_features = preprocessor.preprocess_audio(waveform)

    filename = os.path.basename(file)

    df = pd.DataFrame(columns=['filename'] + [f'mfcc_{i}' for i in range(1, len(ps_features))])

    # Add data to DataFrame
    for i, feature in enumerate(ps_features):
        df.loc[0, f'mfcc_{i+1}'] = feature.mean()

    df['filename'] = filename
    return df

def get_ds_df():
    # Example usage:
    preprocessor = AudioPreprocessor()  # Initialize an instance of AudioPreprocessor
    sids = get_dataframe()

    df_list = []

    for sid in sids:
        audio_df = pd.DataFrame()
        speaker_df = sids[sids['speaker_id'] == sid]

        # Iterate over filtered DataFrame and extract features from each audio file
        for _, row in speaker_df.iterrows():
            audio_file = os.path.join(DATA_DIR, row['audio_filepath'])
            features = extract_features(preprocessor, audio_file)
            audio_df = pd.concat([audio_df, features], ignore_index=True)

        df_list.append(audio_df)
    
    return df_list
