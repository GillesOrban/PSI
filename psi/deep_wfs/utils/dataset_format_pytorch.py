'''
Source: VLADI software
https://github.com/GillesOrban/VLADI
'''
# loading the dataset into splits for training and validation

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import math

"""
    container sizes:
        - self.data_dict['psfs_1']: (nb_samples, 512, 640)
        - self.data_dict['psfs_2']: (nb_samples, 512, 640)
        - self.data_dict['zernike_coeff']: (nb_samples, nb_modes)
    __getitem__(id) returns:
        - sample = datasetObj[id]
        - sample['image']: (1, 512, 640)
        - sample['zernike']: (nb_modes)
        - it returns item after applying self.transform[]
"""


class psf_dataset(Dataset):

    def __init__(self, dataset, data_info, transform=None):

        self.transform = transform
        self.n_channels = data_info['channels']
        self.wavelength = data_info['wavelength'] # in nm
        self.size = int(float(data_info['nb_samples']))

        self.data_dict = {}

        # for inference, zernike_coefficients are not required
        if 'zernike_coefficients' in dataset.keys():
            self.data_dict['zernike_coeff'] = torch.as_tensor(dataset['zernike_coefficients'][:self.size], dtype=torch.float32)

        if self.n_channels == 2:
            self.data_dict['psfs_1'] = torch.as_tensor(np.array(dataset["psfs_1"][:self.size, :, :], dtype="float32"))
            self.data_dict['psfs_2'] = torch.as_tensor(np.array(dataset["psfs_2"][:self.size, :, :], dtype="float32"))
        else:
            self.data_dict['psfs_1'] = torch.as_tensor(np.array(dataset["psfs_1"][:self.size, :, :], dtype="float32"))

        # print("psf size: ", self.data_dict['psfs_1'].size())
        # print("zernike size: ", self.data_dict['zernike_coeff'].size())

    def __len__(self):
        return self.size

    def __getitem__(self, id):
        if self.n_channels == 2:
            images = torch.cat((self.data_dict['psfs_1'][id].unsqueeze(0),  # unsqueeze to concatenate along channels dimension
                                self.data_dict['psfs_2'][id].unsqueeze(0)),
                               dim=0)
        else:
            images = self.data_dict['psfs_1'][id].unsqueeze(0)

        sample = {"image": images.clone()}
        if 'zernike_coeff' in self.data_dict.keys():
            zernike = self.data_dict['zernike_coeff'][id]
            zernike = (zernike / self.wavelength) * 2 * math.pi  # from nm to radians

            sample["zernike"] = zernike

        if self.transform:
            # if id==0:
            #     print("before [0][0]: ", sample['image'][0][0][0])
            #     print("fl")

            sample = self.transform(sample)
            
            # if id==0:
            #     print("after [0][0]: ", sample['image'][0][0][0])


        return sample


def splitDataLoader(dataset, conf, device, split=[0.9, 0.1],
                    shuffle=True, mseed=None):

    indices = list(range(len(dataset)))
    if shuffle:
        if mseed is not None:
            np.random.seed(mseed)
        np.random.shuffle(indices)

    svalid = int(split[1] * len(dataset))
    train_indices, val_indices = indices[svalid:], indices[:svalid]

    train_sampler, val_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)

    if "cuda" in str(device):
        pin_mem = True
    else:
        pin_mem = False

    train_dataloader = DataLoader(dataset,
                                  batch_size=conf["batch_size"],
                                  num_workers=conf["nb_workers"],
                                  sampler=train_sampler,
                                  pin_memory=pin_mem)
    val_dataloader = DataLoader(dataset,
                                batch_size=conf["batch_size"],
                                num_workers=conf["nb_workers"],
                                sampler=val_sampler,
                                pin_memory=pin_mem)

    return train_dataloader, val_dataloader
