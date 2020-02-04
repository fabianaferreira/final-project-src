import numpy as np
from datetime import datetime
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

TEMPLATE_FIGNAME = 'confusion_matrix_' + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
PLOT_DIR = './Plots/'
CONFUSION_MATRIX_DIR = './Confusion_Matrix/'


class ConfusionMatrix:
    def __init__(self, classes, y_true=None, y_pred=None, cm_file=None):
        self.classes = classes
        self.model_name = cm_file
        if cm_file is None:
            self.cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        else:
            self.cm = np.load(CONFUSION_MATRIX_DIR + 'confusion_matrix_model_' + cm_file + '_subset.npy', allow_pickle=True)

    def save_matrix(self, filename=TEMPLATE_FIGNAME):
        np.save(CONFUSION_MATRIX_DIR + filename, self.cm)

    def plot_figure(self, normalize=True, cmap=plt.get_cmap('Reds'),
                    show_annotations=True, fig_size=(12, 8), rotation=45, save_fig=False):
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

        f = plt.figure(figsize=fig_size)

        # Create an axes instance and adjust the subplot size
        ax = f.add_subplot(111)
        ax.set_aspect(1)
        f.subplots_adjust(left=leftmargin / figwidth, right=1 - rightmargin / figwidth, top=0.94, bottom=0.1)

        res = ax.imshow(self.cm, interpolation='nearest', cmap=cmap)
        cbar = f.colorbar(res)
        cbar.ax.tick_params(labelsize=18)

        plt.title(title, fontdict={'fontsize': 22})
        ax.set_xticks(range(len(self.classes)))
        ax.set_yticks(range(len(self.classes)))
        ax.set_xticklabels(self.classes, rotation=rotation, fontdict={'fontsize': 18})
        ax.set_yticklabels(self.classes, fontdict={'fontsize': 18})

        if show_annotations:
            fmt = '.2f' if normalize else 'd'
            thresh = self.cm.max() / 2.
            for i, j in itertools.product(range(self.cm.shape[0]), range(self.cm.shape[1])):
                ax.text(j, i, format(self.cm[i, j], fmt),
                        fontdict={'fontsize': 16},
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="white" if self.cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Classe Alvo', fontdict={'fontsize': 18})
        plt.xlabel('Classe Predita\nacurácia={:0.4f}; erro de classificação={:0.4f}'.format(accuracy, misclass),
                   fontdict={'fontsize': 18})
        plt.grid(b=False)

        if save_fig:
            plt.savefig(PLOT_DIR + 'confusion_matrix_model' + self.model_name + '_subset.svg', format='svg', bbox_inches='tight')
