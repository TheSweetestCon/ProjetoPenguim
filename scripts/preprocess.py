import pandas as pd

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.drop(columns=['Unnamed: 0'])

    num_cols = ['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']
    for col in num_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    df['sex'].fillna(df['sex'].mode()[0], inplace=True)

    df = pd.get_dummies(df, columns=['island','sex'], drop_first=True)
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    preprocess('./data/penguins.csv', './data/penguins_preprocessed.csv')