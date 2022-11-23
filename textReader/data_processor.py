import numpy as np
import os
from image_processor import plot
from collections import namedtuple
import time

Sample = namedtuple('Sample', 'label, path')

class DataProcessor:
    def __init__(self, train_test_split: np.double):
        self.data_path = 'data'
        #115320 as number of samples in dataset
        self.samples = np.empty(115320, dtype=Sample)
        self.train_test_split = train_test_split

        with open(os.path.join(self.data_path, 'gt', 'words.txt')) as f:
            i = 0
            for line in f:
                line = line.strip()
                if not line or line[0] == '#':
                    pass
                else:
                    line = line.split(' ')
                    path = line[0].split('-')
                    path = os.path.join(
                        self.data_path, 'img',
                        path[0],
                        f'{path[0]}-{path[1]}',
                        f'{path[0]}-{path[1]}-{path[2]}-{path[3]}.png'
                    )
                    label = ''.join(line[8:])
                    self.samples[i] = Sample(label, path)
                    i += 1

        self.train_set = self.samples[:int(len(self.samples) * self.train_test_split)]
        self.test = self.samples[int(len(self.samples) * self.train_test_split):]


def main():
    DataProcessor(.9)

if __name__ == '__main__':
    main()