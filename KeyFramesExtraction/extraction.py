import cv2
import numpy as np
import math
import imageio
import glob2 as glob
import os
import shutil
import logging
import time
from DE import *
from tqdm import tqdm


pathToVideos = "/home/fabiana/Desktop/projeto-final-src/Crawler/Videos/"
resultPath = "/home/fabiana/Desktop/projeto-final-src/KeyFramesExtraction/Result/"
n_frames = [5, 10, 15]

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] - %(message)s',
                    filename="/home/fabiana/Desktop/projeto-final-src/KeyFramesExtraction/keyframes.log",
                    filemode='a')


def getAllDirectories(pathToDirectory):
    '''
        Function that gets all the subdirectories of a folder

        Args:
            path: Path to root directory

        Returns: List of all the complete paths to files
    '''

    files = []
    files = glob.glob(pathToDirectory + '*/', recursive=True)
    return files


def getAllFiles(pathToFiles):
    '''
        Function that gets all the filenames from a directory

        Args:
            path: Path to directory

        Returns: List of all the complete paths to files
    '''
    files = []
    files = glob.glob(pathToFiles + '*/*/', recursive=True)
    return files


def extractFramesToFile(pathToFile, dest):

    vidcap = cv2.VideoCapture(pathToFile)
    frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success = True

    for i in range(0, frames):
        success, image = vidcap.read()
        if success:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(dest + "frame" + str(i) + ".jpg",
                        gray)     # save frame as JPEG file
            if cv2.waitKey(10) == 27:                     # exit if Escape is hit
                break
    # When everything done, release the capture
    vidcap.release()
    cv2.destroyAllWindows()


def processSign(framesPath, n):

    stop_iteration = 10
    number_candidates = 10
    scale = 0.9
    crossover = 0.6

    framesQnt = len(glob.glob(framesPath + "*.jpg"))
    if (framesQnt < 1):
        print("Corrupted video:" + framesPath)
        return -1

    de = DifferentialEvolution(
        n, scale, crossover, number_candidates, stop_iteration)

    print(f"Starting routine for file {framesPath} with n = {n}")
    print("Quantity of frames: " + str(framesQnt))

    de.initialize_NP(framesQnt, framesPath)
    for GENERATION in range(stop_iteration):
        for j in range(number_candidates):
            de.mutation(j, framesQnt, framesPath)
            de.crossover(j, framesPath)
            de.selection(j)

    best = de.bestParent()
    keys = framesPath.rsplit('/', 3)[:-1]

    if (len(best[:-1]) < n):
        logging.warning(
            f"Video for word {keys[2]} got less than {n} key frames")

    result = os.path.join(resultPath, f"{n}/{keys[1]}/{keys[2]}/")
    os.makedirs(result)
    for frame_number in best[:-1]:
        shutil.copyfile(framesPath + "frame" + str(frame_number) +
                        ".jpg", result + "frame" + str(frame_number) + ".jpg")


directoriesList = getAllDirectories(pathToVideos)
filesList = getAllFiles(pathToVideos)

for n in n_frames:
    print(f"Starting routine for keyframes extraction with n = {n}")
    logging.info(f"Starting routine for keyframes extraction with n = {n}")
    start_time = time.time()
    for f in tqdm(filesList, total=len(filesList)):
        processSign(f, n)
    print(
        f"Execution time for routine with n = {n} -> {time.time() - start_time}")
    logging.info(
        f"Execution time for routine with n = {n} -> {time.time() - start_time}")
