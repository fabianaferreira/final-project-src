from Extractor import *
from glob import glob
import pandas as pd
import re
from tqdm import tqdm
import logging


logging.basicConfig(format='%(asctime)s [%(levelname)s] - %(message)s',
                    filename="/home/fabiana/Desktop/projeto-final-src/Classifier/InceptionV3Features.log",
                    filemode='a')

def buildSequence(frameList):
    sequence = []
    for image in frameList:
        features = model.extract(image)
        sequence.append(features)
    return np.array(sequence)


proc_csv = pd.read_pickle('dataset_with_file_list.pkl')

logging.info("Starting routine for features extraction with InceptionV3")

model = Extractor()

proc_csv.set_index('palavra', inplace=True)
folder = '/home/fabiana/Desktop/projeto-final-src/Classifier/InceptionV3_Features'

for n in ['5', '10', '15']:
    print('Number of keyframes: ' + n)
    for palavra, frameList in tqdm(proc_csv[f'files_list_{n}'].iteritems(), total=len(proc_csv)):
        if (len(frameList) < int(n)):
            # print(f"Word {palavra} got less than {n} key frames")
            logging.warning(f"Word {palavra} got less than {n} key frames")

        seq = buildSequence(frameList)
        np.save(f'{folder}/{n}/{palavra}', seq)
