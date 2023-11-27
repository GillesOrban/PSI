'''
copied from 'inlab_deep_learning_fpwfs'
'''
import numpy as np
import h5py


def save_to_hdf5(filename, psfs, zernike_coeff, config):
    """
    Example showing how the training datasets should be saved in the hdf5 format.
    """
    with h5py.File(filename, mode="w") as hdf:
        # Save the Zernike coefficients:
        hdf.create_dataset("zernike_coefficients", data=zernike_coeff)
        # Save the PSFs:
        hdf.create_dataset("psfs", data=psfs,
                           compression="gzip", compression_opts=4)
        # Add attributes:
        hdf["zernike_coefficients"].attrs['unit'] = config["zernike_unit"]
        hdf["psfs"].attrs['defocus'] = config["defocus"]  # in nm
        hdf.attrs['seed'] = config["zernike_seed"]
        # hdf.attrs['nb_samples'] = zernike_coeff.shape[0]

def read_data(filename, dataset_size):
    """
    Example showing how the hdf5 datasets can be read.
    """
    with h5py.File(filename, 'r') as hf:
        # Putting the dataset as tensors into a dictionary:
        zern_coeffs = np.array(hf['zernike_coefficients'][:dataset_size])
        psfs_in = np.array(hf["psfs"][0, :dataset_size, :, :])
        psfs_out = np.array(hf["psfs"][1, :dataset_size, :, :])

    return zern_coeffs, psfs_in, psfs_out


if __name__ == '__main__':
    fname = "./dataset_example.hdf5"  # indicate the path and the name for the dataset file:
    conf = {"n_modes": 10,
            "n_samples": 10,
            "n_channels": 2,
            "pupil_size": 64,
            "det_size": 64,
            "zernike_seed": 0,
            "zernike_unit": "nm",
            "defocus": 440
            }

    toy_psfs = np.zeros((conf["n_channels"], conf["n_samples"], conf["det_size"], conf["det_size"]))
    toy_zern = np.zeros((conf["n_samples"], conf["n_modes"]))
    toy_zern_poly = list(range(4, conf["n_modes"] + 4))  # 4: defocus mode
    toy_pupil = np.zeros((conf["pupil_size"], conf["pupil_size"]))

    save_to_hdf5(filename=fname,
                 psfs=toy_psfs,
                 zernike_coeff=toy_zern,
                 config=conf)

    zern_coeff_read, psfs_in_read, psfs_out_read = read_data(filename=fname,
                                                             dataset_size=conf["n_samples"])

    assert zern_coeff_read.all() == toy_zern.all()
    assert psfs_in_read.all() == toy_psfs[0].all()
    assert psfs_out_read.all() == toy_psfs[1].all()
