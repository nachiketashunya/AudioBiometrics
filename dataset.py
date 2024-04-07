import os
import pandas as pd
import torchaudio
from preprocess import AudioPreprocessor  # Import the AudioPreprocessor class from your preprocessing module
from get_df import get_dataframe
from sklearn.model_selection import train_test_split

DATA_DIR = "data/svarah/audio"

def extract_features(self, preprocessor, file, speaker_id):
    # Load the audio file
    waveform, _ = torchaudio.load(file)

    # Preprocess audio using the provided preprocessor
    ps_features = preprocessor.preprocess_audio(waveform)

    filename = os.path.basename(file)

    df = pd.DataFrame(columns=['speaker_id'] + ['filename'] + [f'mfcc_{i}' for i in range(1, len(ps_features))])

    # Add data to DataFrame
    for i, feature in enumerate(ps_features):
        df.loc[0, f'mfcc_{i+1}'] = feature.mean()

    df['speaker_id'] = speaker_id
    df['filename'] = filename
    return df

def get_mapping():
    sub_df, sids = get_dataframe()()
    return {val: idx + 1 for idx, val in enumerate(set(sids))}

def get_ds_df():
    # Example usage:
    preprocessor = AudioPreprocessor()  # Initialize an instance of AudioPreprocessor
    sub_df, sids = get_dataframe()()

    mapping = get_mapping()

    df_list = []

    for sid in sids:
        print(sid)

        audio_df = pd.DataFrame()
        speaker_df = sub_df[sub_df['speaker_id'] == sid]
        speaker_id = mapping.get(sid)

        # Iterate over filtered DataFrame and extract features from each audio file
        for _, row in speaker_df.iterrows():
            audio_file = os.path.join(DATA_DIR, row['audio_filepath'])
            features = extract_features(preprocessor, audio_file, speaker_id)
            audio_df = pd.concat([audio_df, features], ignore_index=True)

        df_list.append(audio_df)
    
    train_dfs = []
    test_dfs = []

    for df in df_list:
        train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)
        train_df = pd.DataFrame(train)
        test_df = pd.DataFrame(test)

        train_dfs.append(train_df)
        test_dfs.append(test_df)
        
    return train_dfs, test_dfs
    
def get_enroll_ds():
    sub_df, sids = get_dataframe()()
    mapping = get_mapping()

    enroll_df = sub_df[sub_df['speaker_id']=="281474976895472"]
    enroll_id = max(mapping.values()) + 1

    return enroll_id, enroll_df
