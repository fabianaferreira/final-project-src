import pandas as pd
import matplotlib.pyplot as plt


def plot_accuracy_loss(model_name, metric='accuracy', save_fig=False):
    history_file = './Metrics/history_' + model_name + '_subset.csv'
    h = pd.read_csv(history_file)

    if metric == 'accuracy':
        title = 'Acurácia do Modelo'
        y_label = 'Acurácia'
    elif metric == 'loss':
        title = 'Loss do Modelo'
        y_label = 'Loss'
    elif metric == 'top_2_accuracy':
        title = 'Acurácia Top-2 do Modelo'
        y_label = 'Acurácia'
    elif metric == 'top_3_accuracy':
        title = 'Acurácia Top-3 do Modelo'
        y_label = 'Acurácia'

    f = plt.figure(figsize=(12, 8))
    plt.plot(h[metric])
    plt.plot(h['val_' + metric])
    plt.title(title, fontdict={'fontsize': 24})
    plt.ylabel(y_label, fontdict={'fontsize': 20})
    plt.xlabel('Época', fontdict={'fontsize': 20})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(['Treino', 'Validação'], loc=('upper right' if metric == 'loss' else 'lower right'), fontsize=18)
    plt.grid(which='major')

    if save_fig:
        plt.savefig('./Plots/' + metric + '_' + model_name + '.svg', dpi=500, format='svg', bbox_inches='tight')
