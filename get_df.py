import pandas as pd

CSV_FILE = "data/svarah/meta_speaker_stats.csv"

def extract_speaker_id(filepath):
    parts = filepath.split('/')
    speaker_id = parts[1].split('_')[0]
    return speaker_id

def get_dataframe():
    csv = pd.read_csv(CSV_FILE)

    audio_csv = pd.DataFrame(csv.loc[:, 'audio_filepath'])
    # Apply the function to all rows of the DataFrame
    audio_csv['speaker_id'] = audio_csv['audio_filepath'].apply(extract_speaker_id)

    audio_csv['speaker_id'] = audio_csv['audio_filepath'].str.split('_').str[0].str.split('/').str[1]
    # Group by speaker ID and count occurrences
    speaker_counts = audio_csv.groupby('speaker_id').size()
    # Filter speaker IDs with more than 15 samples
    subsampled_sid = speaker_counts[speaker_counts > 15].index.tolist()
    # Filter DataFrame to keep only rows with matching speaker IDs
    sub_df = audio_csv[audio_csv['speaker_id'].isin(subsampled_sid)]
   
    return sub_df, subsampled_sid