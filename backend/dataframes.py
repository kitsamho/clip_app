import numpy as np
import pandas as pd

def results_to_dataframe(probs, texts):
    df_result = pd.DataFrame(probs[0].detach().numpy(), texts).reset_index()
    df_result.columns=['labels', 'probabilities']
    return df_result

def concatenate_dataframes(df_embeddings, df_user_embedding):
    df = pd.concat([df_embeddings, df_user_embedding], ignore_index=True)
    return df


def get_umap_dataframe(model, embeddings):
    arrays = [i.reshape(1, -1) for i in embeddings]
    df_umap = pd.DataFrame(model.fit_transform(np.concatenate(arrays, axis=0)))
    df_umap.columns = ['x', 'y']
    return df_umap