import torch
from sklearn.metrics import accuracy_score, classification_report

def evaluate_dl(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print(classification_report(all_labels, all_preds))
    
def compute_val_accuracy(model, dataloader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x).squeeze()
            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).long()

            preds.extend(pred.cpu().numpy())
            labels.extend(y.numpy())

    model.train()
    return accuracy_score(labels, preds)

