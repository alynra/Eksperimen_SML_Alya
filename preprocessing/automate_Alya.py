import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_data(df):
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                    'Friends_circle_size', 'Post_frequency']
    categorical_cols = ['Stage_fear', 'Drained_after_socializing', 'Personality']

    df_cleaned = df.drop_duplicates()

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_cleaned[numeric_cols]),
        columns=numeric_cols,
        index=df_cleaned.index
    )

    df_scaled = pd.concat([df_scaled, df_cleaned[categorical_cols]], axis=1)

    le = LabelEncoder()
    df_encoded = df_scaled[numeric_cols].copy()
    for col in categorical_cols:
        df_encoded[col] = le.fit_transform(df_scaled[col])

    return df_encoded

def main():
    df = pd.read_csv("personality_raw.csv")
    df_processed = preprocess_data(df)
    df_processed.to_csv("preprocessing/personality_preprocessing.csv", index=False)
    print("Preprocessing selesai dan disimpan.")

if __name__ == "__main__":
    main()
