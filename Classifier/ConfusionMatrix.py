import numpy as np
from datetime import datetime
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

TEMPLATE_FIGNAME = 'confusion_matrix_' + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
PLOT_DIR = './Plots/'


class ConfusionMatrix:
    def __init__(self, y_true, y_pred, classes):
        self.classes = classes
        self.cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

    def plotFigure(self, normalize=False, cmap=plt.get_cmap('Reds'),
                   show_annotations=False, figsize=(12, 8), rotation=45, showfig=True, figname=TEMPLATE_FIGNAME):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            self.cm = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
            title = 'Matriz de Confusão Normalizada'
        else:
            title = 'Matriz de Confusão'

        accuracy = np.trace(self.cm) / float(np.sum(self.cm))
        misclass = 1 - accuracy

        # Calculate chart area size
        leftmargin = 0.5  # inches
        rightmargin = 0.5  # inches
        categorysize = 0.5  # inches
        figwidth = leftmargin + rightmargin + (len(self.classes) * categorysize)

        f = plt.figure(figsize=figsize)

        # Create an axes instance and adjust the subplot size
        ax = f.add_subplot(111)
        ax.set_aspect(1)
        f.subplots_adjust(left=leftmargin / figwidth, right=1 - rightmargin / figwidth, top=0.94, bottom=0.1)

        if not normalize:
            res = ax.imshow(self.cm, interpolation='nearest', cmap=cmap)
            plt.colorbar(res)

        plt.title(title, fontdict={'fontsize': 18})
        ax.set_xticks(range(len(self.classes)))
        ax.set_yticks(range(len(self.classes)))
        ax.set_xticklabels(self.classes, rotation=rotation, ha='right', fontdict={'fontsize': 12})
        ax.set_yticklabels(self.classes, fontdict={'fontsize': 12})

        if show_annotations:
            fmt = '.2f' if normalize else 'd'
            thresh = self.cm.max() / 2.
            for i, j in itertools.product(range(self.cm.shape[0]), range(self.cm.shape[1])):
                ax.text(j, i, format(self.cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if self.cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Classe Alvo', fontdict={'fontsize': 14})
        plt.xlabel('Classe Predita\nacurácia={:0.4f}; erro de classificação={:0.4f}'.format(accuracy, misclass),
                   fontdict={'fontsize': 14})

        if showfig:
            plt.show()
            plt.close(f)
        else:
            plt.savefig(PLOT_DIR + figname + '.eps', format='eps', dpi=1200, bbox_inches='tight')
            plt.close(f)