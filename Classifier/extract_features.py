from Extractor import *
from glob import glob
import pandas as pd
import re
from tqdm import tqdm


def getFrameNumber(path):
    return int(re.search(r".*\/frame(\d+)\.jpg", path).group(1))


def getFiles(path):
    files = list(glob(path + "/*.jpg"))
    files.sort(key=getFrameNumber)
    return files


def buildSequence(frameList):
    sequence = []
    for image in frameList:
        features = model.extract(image)
        sequence.append(features)
    return np.array(sequence)

proc_csv = pd.read_pickle('dataset_with_file_list.pkl')
model = Extractor()

proc_csv.set_index('palavra', inplace=True)
folder = '/home/fabiana/Desktop/projeto-final-src/Classifier/InceptionV3_Features'

for n in ['5', '10', '15']:
    print('Number of keyframes: ' + n)
    for palavra, frameList in tqdm(proc_csv[f'files_list_{n}'].iteritems(), total=len(proc_csv)):
        seq = buildSequence(frameList)
        np.save(f'{folder}/{n}/{palavra}', seq)
