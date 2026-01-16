from sklearn.metrics import accuracy_score, classification_report

def evaluate(model, X, y):
    preds = model.predict(X)
    print("Accuracy:", accuracy_score(y, preds))
    print(classification_report(y, preds))
