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


class Noise(object):
    def __init__(self, signal, bckg):
        self.signal = signal
        self.bckg = bckg

    def __call__(self, sample):
        assert len(sample['image'].shape) == 3

        for i in range(sample['image'].shape[0]):
            image = sample['image'][i]
            norm = torch.sum(image)
            if self.bckg == 0 :
                noisy_image = torch.poisson(image / norm * self.signal)
            else:
                # print('[debug] : adding background')
                background = torch.as_tensor(self.bckg)
                background_noise = torch.poisson(background + image*0) - background
                noisy_image = torch.poisson(image / norm * self.signal) + background_noise
            sample['image'][i] = noisy_image
        
        return sample