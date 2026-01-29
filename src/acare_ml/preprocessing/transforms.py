import pandas as pd


def normalize_signal(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df_norm = df.copy()
    df_norm[columns] = (df[columns] - df[columns].mean()) / df[columns].std()
    return df_norm
