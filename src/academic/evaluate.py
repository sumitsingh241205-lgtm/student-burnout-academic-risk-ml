from sklearn.metrics import accuracy_score, f1_score, classification_report

def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("Test Accuracy:", acc)
    print("Macro F1:", macro_f1)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))