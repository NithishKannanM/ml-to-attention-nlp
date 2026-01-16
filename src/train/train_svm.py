from src.data.dataset import load_and_split
from src.features.vectorizer import get_tfidf_vectorizer
from src.models.svm_model import build_svm
from src.evaluate.metrics import evaluate
import joblib
import os



X_train, X_val, y_train, y_val = load_and_split("dataset/IMDB Dataset.csv")

vectorizer = get_tfidf_vectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

model = build_svm()
model.fit(X_train_vec, y_train)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/svm.pkl")
joblib.dump(vectorizer, "models/tfidf.pkl")

evaluate(model, X_val_vec, y_val)
