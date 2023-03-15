import pandas as pd

def results_to_dataframe(probs, texts):
    df_result = pd.DataFrame(probs[0].detach().numpy(), texts).reset_index()
    df_result.columns=['labels', 'probabilities']
    return df_result