import matplotlib.pyplot as plt

def plot_training(train_losses, val_accuracies=None, title="Training Curve"):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")

    if val_accuracies:
        plt.plot(epochs, val_accuracies, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.show()
    