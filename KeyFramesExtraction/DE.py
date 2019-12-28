from skimage.measure import compare_ssim as ssim
import cv2
import random

class DifferentialEvolution:
    def __init__(self, totalKeyFrames, scale, probability, candidates, stop):
        self.__n_frames = totalKeyFrames
        self.__f = scale
        self.__cr = probability
        self.__n_candidates = candidates
        self.__stop = stop
        # Population matrix.
        self.NP = []
        # Mutation vector.
        self.MV = []
        # Trail vector.
        self.TV = []

    # Calculate ASSIM for a chromosome.
    def getASSIM(self, KF, filepath):
        ssim_sum = 0
        for i in range(0, self.__n_frames - 1):
            try:
                im1 = cv2.imread(filepath + "frame" +
                                    str(KF[i]) + ".jpg", 0)
                im2 = cv2.imread(filepath + "frame" +
                                    str(KF[i+1]) + ".jpg", 0)
                ssim_sum += ssim(im1, im2)
            except:
                print(i, KF[i], KF[i+1])
                raise
        return ssim_sum/(self.__n_frames)

    # INITIALISATION : Generates population NP of 10 parent vectors (and ASSIMs).
    def initialize_NP(self, maxNumberOfFrames, filepath):
        for i in range(self.__n_candidates):
            self.NP.append(sorted(random.sample(
                range(0, maxNumberOfFrames), self.__n_frames)))
            self.NP[-1].append(self.getASSIM(self.NP[-1], filepath))

    # MUTATION
    def mutation(self, parent, maxNumberOfFrames, filepath):
        R = random.sample(self.NP, 2)
        # cleaning list
        self.MV[:] = []
        MV_value = 0
        for i in range(self.__n_frames):
            parent_value = self.NP[parent][i]
            mutation_value = self.__f*(R[0][i] - R[1][i])
            MV_value = int(parent_value + mutation_value)
            if(MV_value < 1):
                self.MV.append(1)
            elif(MV_value > maxNumberOfFrames-1):
                self.MV.append(maxNumberOfFrames-1)
            else:
                self.MV.append(MV_value)
        self.MV.sort()
        self.MV.append(self.getASSIM(self.MV, filepath))
        if(len(self.MV) == 0):
            print("problem with MV")

    # CROSSOVER (uniform crossover with Cr = 0.6).
    def crossover(self, index, filepath):
        for j in range(self.__n_frames):
            if(random.uniform(0, 1) < self.__cr):
                self.TV.append(self.MV[j])
            else:
                self.TV.append(self.NP[index][j])
        self.TV.sort()
        self.TV.append(self.getASSIM(self.TV, filepath))
        if(len(self.TV) == 0):
            print("problem with TV")

    # SELECTION : Selects offspring / parent based on lower ASSIM value.
    def selection(self, index):
        if(self.TV[-1] < self.NP[index][-1]):
            self.NP[index] = self.TV
        self.TV = []

    # bestParent returns the parent with then minimum ASSIM value.
    def bestParent(self):
        population = self.NP
        Min_ASSIM_value = population[0][-1]
        Best_Parent_Index = population[0]
        for parent in population:
            if (parent[-1] < Min_ASSIM_value):
                Min_ASSIM_value = parent[-1]
                Best_Parent_Index = parent
        return Best_Parent_Index
