import math
import matplotlib.pyplot as plt
import numpy as np

import itertools

def show_losses(histories, acc='acc'):
    plt.figure(figsize=(10,10))
    #plt.ylim(bottom=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error by Epoch')
    colors=[]
    do_acc=False
    for label,loss in histories:
        color = tuple(np.random.random(3))
        colors.append(color)
        l = label
        vl= label+" validation"
        if acc in loss.history:
            l+=' (%s %2.4f)'% (acc, loss.history[acc][-1])
            do_acc = True
        if 'val_'+acc in loss.history:
            vl+=' (val_%s %2.4f)'% (acc,loss.history['val_'+acc][-1])
            do_acc = True
        plt.plot(loss.history['loss'], lw=4, label=l, color=color)
        if 'val_loss' in loss.history:
            plt.plot(loss.history['val_loss'], lw=4, ls='dashed', label=vl, color=color)
    plt.legend(loc='best')
    #plt.yscale('log')
    plt.show()
    
    if not do_acc: return
    plt.figure(figsize=(10,10))
    plt.xlabel('Epoch')
    plt.ylabel(acc)
    for i,(label,loss) in enumerate(histories):
        color = colors[i]
        if acc in loss.history:
            plt.plot(loss.history[acc], lw=4, label=label, color=color)
        if 'val_'+acc in loss.history:
            plt.plot(loss.history['val_'+acc], lw=4, ls='dashed', label=label + ' validation', color=color)
    plt.legend(loc='best')
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
