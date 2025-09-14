import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv('breast-cancer.csv')
    df = df.drop(columns=['id'])
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    corr = df.corr()
    first_row = corr.iloc[0]
    sorted_row = first_row.abs().sort_values(ascending=False)

    ## Select top 5 features correlated with diagnosis: concave points_worst, perimeter_worst, concave points_mean, radius_worst, perimeter_mean
    features = ['concave points_worst', 'perimeter_worst', 'concave points_mean', 'radius_worst', 'perimeter_mean']
    X = df[features]
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize Features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test