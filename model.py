
from sklearn.mixture import GaussianMixture
import joblib
import os
from .get_df import get_dataframe
from .dataset import get_ds_df

SAVE_DIR = "models/"

# Function to train GMM for each speaker
def train_gmm_for_speakers(dfs, subsampled_sid):
    for i in range(len(subsampled_sid)):
        speaker_id = subsampled_sid[i]  # Assuming 'speaker_id' is in the DataFrame
        mfcc_features = dfs[i].drop(columns=['filename'])  # Assuming 'speaker_id' is a column to drop

        # Train GMM
        gmm = GaussianMixture(n_components=5)  # Adjust the number of components as needed
        gmm.fit(mfcc_features)

        # Save trained GMM model
        model_filename = os.path.join(SAVE_DIR, f"{speaker_id}_gmm_model.pkl")

        joblib.dump(gmm, model_filename)
        
        print(f"GMM model saved for speaker ID {speaker_id} as {model_filename}")


df_list = get_ds_df()
sids = get_dataframe()

# Train GMM for each speaker and save the models
train_gmm_for_speakers(df_list, sids)