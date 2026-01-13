import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Val')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Accuracy
    ax2.plot(history.history['accuracy'], label='Train')
    ax2.plot(history.history['val_accuracy'], label='Val')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.show()

def plot_cm(model, dataset, class_names=['Different', 'Same']):
    y_true, y_pred = [], []
    for X_batch, y_batch in dataset:
        preds = (model.predict(X_batch) > 0.5).astype(int)
        y_pred.extend(preds.flatten())
        y_true.extend(y_batch.astype(int))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()