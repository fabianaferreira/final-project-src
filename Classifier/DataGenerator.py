from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
import numpy as np


class DataGenerator(Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx *
                                       self.batch_size: (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx+1) * self.batch_size]

        return (
            np.array([load_image(file_name) for file_name in batch_x]),
            np.array(batch_y)
        )

def load_image(image_path):
    img = image.load_img(
        image_path, color_mode='grayscale')
    img = image.img_to_array(img)/255.0
    return img
    # img = np.squeeze(image.img_to_array(img))/255.0
    # img /= 127.5
    # img -= 1.
