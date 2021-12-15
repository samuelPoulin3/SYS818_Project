import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import roc_curve, auc

# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    
    plt.figure(figsize=(12,12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if classes!=None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    
def plot_roc(y_test, y_pred, text_label):

    for key in list(y_test.keys()):
        fpr, tpr, _ = roc_curve(y_test[key][:, 0], y_pred[key][:, 0])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(7,7))
        lw=2
        alpha=1
        plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
        plt.plot(fpr, tpr, 
                label= text_label+r' (AUC = %0.2f )' % (roc_auc),lw=lw, alpha=alpha)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Male-Female Classification')
    plt.legend(loc="lower right")
    plt.show()