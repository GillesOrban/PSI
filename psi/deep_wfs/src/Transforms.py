import torch

class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):

        for i in range(sample['image'].shape[0]):
            if torch.min(sample['image'][i]) < 0:
                # Shift values above 0:
                sample['image'][i] -= torch.min(sample['image'][i])
            sample['image'][i] = self.minmax(torch.sqrt(sample['image'][i]))
        return sample

    def minmax(self, array):
        a_min = torch.min(array)
        a_max = torch.max(array)

        if abs(a_max - a_min) > 0:
            array_norm = (array - a_min) / (a_max - a_min)
        else:
            array_norm = torch.zeros(array.shape)
        return array_norm