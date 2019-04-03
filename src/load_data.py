import pandas as pd


def load_data(path):
    train_df = pd.read_csv(path)
    X_data = train_df['comment_text'].astype(str).values.tolist()
    y_data = [int(target >= 0.5) for target in train_df['target']]
    return X_data, y_data
