from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from glob import glob
import re

path = "/home/fabiana/Desktop/projeto-final-src/Crawler/Videos/"
df = pd.read_csv('../Annotations/processedAnnotations_no_corrupted_videos.csv')
count = df.groupby(['CM']).count()['palavra']
threshold = 20
frames = 10
cnn_dataset_path = "/home/fabiana/Desktop/projeto-final-src/Classifier/datasets/CNN"


def getFrameNumber(path):
    return int(re.search(r".*\/frame(\d+)\.jpg", path).group(1))


def getFiles(path):
    files = list(glob(path + "/*.jpg"))
    files.sort(key=getFrameNumber)
    return files


def replaceClass(cg):
    quantity = count[cg]
    if (quantity <= threshold):
        return 'others'
    else:
        return cg


df['classe'] = df['CM'].map(replaceClass)
df['classe'] = df['classe'].astype('category')

df = df[['palavra', 'classe']]
df['first_letter'] = df['palavra'].map(lambda x: x[0])
df['folder_path'] = path + df['first_letter'] + \
    '/' + df['palavra'] + f'/DEResult_{frames}/'
df['files_list'] = df['folder_path'].map(getFiles)


X_train, X_test, y_train, y_test = train_test_split(df['files_list'], df['classe'], stratify=df['classe'],
                                                    test_size=0.2, random_state=0)
y_train = np.repeat(y_train.cat.codes.values, X_train.map(len))
y_test = np.repeat(y_test.cat.codes.values, X_test.map(len))
X_test_sum = np.array(X_test.sum())
X_train_sum = np.array(X_train.sum())

np.save(cnn_dataset_path + '/X_train.npy', X_train_sum)
np.save(cnn_dataset_path + '/X_test.npy', X_test_sum)
np.save(cnn_dataset_path + '/y_train.npy', y_train)
np.save(cnn_dataset_path + '/y_test.npy', y_test)
