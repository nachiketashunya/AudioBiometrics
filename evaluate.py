import os
import joblib
from dataset import get_ds_df

class SpeakerVerification:
    def __init__(self, models_dir="/content/models/", threshold=-38000000):
        self.models_dir = models_dir
        self.threshold = threshold

    def load_models(self):
        return os.listdir(self.models_dir)

    def calculate_score(self, gmm, features):
        return gmm.score_samples(features).mean()

    def calculate_eer(self, test_dfs, speakers_models):
        model_ids = []
        eers = []
        for i, gdf in enumerate(test_dfs):
            frr, far = 0, 0
            genuine_speaker_id = gdf['speaker_id'].iloc[0]
            genuine_features = gdf.drop(columns=['filename', 'speaker_id'])

            model_ids.append(genuine_speaker_id)
            gmm = None
            for model in speakers_models:
                if str(genuine_speaker_id) in model:
                    gmm = model
                    break

            if gmm:
                gmm = joblib.load(os.path.join(self.models_dir, gmm))
                score = self.calculate_score(gmm, genuine_features)
                if score < self.threshold:
                    frr += 1

                for j, idf in enumerate(test_dfs):
                    if i != j:
                        imp_speaker_id = idf['speaker_id']
                        imp_features = idf.drop(columns=['filename', 'speaker_id'])
                        iscore = self.calculate_score(gmm, imp_features)
                        if iscore < self.threshold:
                            far += 1

                far = far / (len(test_dfs) - 1)
                eer = (far + frr) / 2
                eers.append(eer)
                print(f"EER for {gmm}: {eer:0.4f}%")
            else:
                print(f"No model found for speaker ID {genuine_speaker_id}")

        return eers

_, test_dfs = get_ds_df()
# Usage
speaker_verifier = SpeakerVerification()
models = speaker_verifier.load_models()
eers = speaker_verifier.calculate_eer(test_dfs, models)
