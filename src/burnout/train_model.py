import joblib
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_burnout_data

def train_model():

    X_train, X_test, y_train, y_test, feature_columns = load_burnout_data()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/burnout_final_model.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")

    return model