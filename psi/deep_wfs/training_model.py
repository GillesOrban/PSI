import os
import time
import numpy as np
import json
import torch
import torch.optim as optim
import psi.deep_wfs.src.Criterion as criterion
import psi.deep_wfs.src.Resnet as resnet
import psi.deep_wfs.utils.dataset_format_pytorch as datapp
import psi.deep_wfs.utils.read_data as rt
from psi.deep_wfs.utils.dataset_format_pytorch import normalization
import psi.deep_wfs.utils.training_tools as utils
from psi.helperFunctions import LazyLogger
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# rt = readTools()
# fmt = _fmt()

class dataTrain:
    def __init__(self, logger=LazyLogger('deep_train')):
        self.logger = logger
        pass

    def setup(self, conf_file=None, **kwargs):
        self.setConfig(conf_file, **kwargs)
        # self._config = _config
        # self._data_info = _data_info
        # if not 'nb_modes' in self._data_info:
        #     print('Warning: data info incomplete, setting nb_modes from config')
        #     self._data_info['nb_modes'] = self._config['nb_modes']
        #     self._data_info['channels'] = 1
        #     self._data_info['wavelength'] = 1
        dd = self.config['model_dir']
        self.logger.info('Setting TensorBoard log dir to {0}'.format(dd))
        self._tb_writer = SummaryWriter(dd)

    def trainModel(self, showTrainingCurve=False, useTensorBoard=True):
        '''
        setup/setConfig needs to be called before
        '''
        # Reading information about the data (hdf5 attributes):
        self.data_info = {}
        self.dataset_master = {}
        self.dataset_master, self.data_info = rt.read_h5(filename=self.config["training_data_fname"])
        self.logger.info("Data the model will be trained on: ")
        for key, val in self.data_info.items():
            print(f"{key}: {val}")

        # Dumping the data info into a json file:
        self.dumpDataAttrs()

        # Specify the device used (gpu or cpu):
        # if no GPU available goes to local machine (cpu)
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.logger.info("Training on GPU")
            self.logger.info(f"Number of GPUs available:{str(torch.cuda.device_count())}")
            self.logger.info(f"GPU name: {str(torch.cuda.get_device_name(0))}")
        else:
            self.logger.info(f"Training on CPU")

        # Setting the DataLoaders:
        dataset_pp = normalization(self.dataset_master, self.data_info)
        self.generateDataLoaders(dataset_pp)

        # TODO: Loading the weights of the model if specified
        
        # Load the model architecture (without trained weights):
        self.model_to_train = resnet.Net(n_channels_in=self.data_info["channels"],
                                    n_channels_out=self.data_info["nb_modes"],
                                    resnet_archi=self.config["model_name"])
                                    #weights_path=pretrained_weights_path)

        # Add model to TensorBoard 
        if useTensorBoard:
            dataiter = iter(self.dataset_loaders['train'])
            dd = next(dataiter)
            # self._toto_images = images
            # self._toto_labels = labels
            self._tb_writer.add_graph(self.model_to_train, dd['image'])
            # adding a ''projector'' to TensorBoard ?
            # [...] see here : https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

        # Moving the model to GPU(s) if possible:
        if torch.cuda.device_count() > 1:
            self.model_to_train = torch.nn.DataParallel(self.model_to_train)
        self.model_to_train.to(self.train_device)

        # Defining the loss function:
        loss_function = criterion.RMSELoss()

        # Defining the optimizer:
        if self.config["optimizer_name"] == 'Adam':
            optimizer = optim.Adam(params=self.model_to_train.parameters(),
                                lr=float(self.config["learning_rate"]))
        else:
            self.logger.error("Chosen optimizer is not implemented")
            raise ValueError("the optimizer specified is not valid (must be either 'Adam' )")

        # TODO: (review) Defining the scheduler:
        optim_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                            mode='min',
                                                            factor=0.5,
                                                            patience=self.config["lr_patience"])
        # Defining the metrics dictionary:
        metric_dict = {
            'model_dir': self.config["model_dir"],
            'pre-trained_weights_dir': self.config["weights_init_dir"],
            'device': str(self.train_device),
            'optimizer': optimizer.__class__.__name__,
            'criterion': loss_function.__class__.__name__,
            'scheduler': optim_scheduler.__class__.__name__,
            'learning_rate_patience': self.config["lr_patience"],
            'earlystop_patience': self.config["earlystop_patience"],
            'dataset_size': int(float(self.data_info["nb_samples"])),
            'training_size': int(self.split[0] * float(self.data_info["nb_samples"])),
            'validation_size': int(self.split[1] * float(self.data_info["nb_samples"])),
            'neural_network': self.config["model_name"],
            'n_epochs': self.config["n_epochs"],
            'batch_size': self.config["batch_size"],
            'avg_epoch_time': '0m 0s',
            'learning_rate': [],
            'train_loss': [],
            'val_loss': [],
        }

        # Metrics:
        metrics_path = os.path.join(self.config["model_dir"], 'metrics.json')
        # Model paths:
        best_model_path = os.path.join(self.config["model_dir"], 'model.pth')

        # Initialize the EarlyStopping object:
        early_stopping = utils.EarlyStopping(model_path=best_model_path,
                                            patience=self.config["earlystop_patience"],
                                            eps_factor=0)

        since = time.time()
        avg_epoch = 0.0
        n_epochs = self.config["n_epochs"]



        for ep in range(n_epochs):

            self.logger.info('-' * 30)
            epoch_time = time.time()
            self.logger.info(f"({str(ep + 1)}/{str(n_epochs)}) EPOCH")
            # #######################
            # ### Training phase: ###
            # #######################

            self.model_to_train.train()  # Set the model to training mode

            train_loss_batch = 0.
            # Iterate over the training batches:
            for _, sample in enumerate(self.dataset_loaders['train']):
                # Handles GPU support
                inputs = sample['image'].to(self.train_device)

                ground_truth = sample['zernike'].to(self.train_device)

                ##############################################################

                # Zero the parameter gradients
                # The backward() function accumulates gradients -> zero_grad() not to mix up gradients between minibatches
                optimizer.zero_grad()

                # Forward propagation: enable gradient calculation only if train
                with torch.set_grad_enabled(True):
                    # 1. Making prediction:
                    estimation = self.model_to_train(inputs)
                    # 2. Computing the loss for the current batch:
                    loss = loss_function(torch.squeeze(estimation), torch.squeeze(ground_truth))
                    # Backward propagation
                    loss.backward()
                    optimizer.step()

                train_loss_batch += 1 * loss.item() * inputs.size(0)
                del loss  # delete loss object to save memory

            # The loss over one epoch is the mean of the losses over the batches:
            train_loss_epoch = train_loss_batch / self.dataset_size["train"]
            self.logger.info('[{0}/{1}] train loss: {2:.6f}'.format(ep + 1, n_epochs, train_loss_epoch))

            # Update metrics
            metric_dict['train_loss'].append(train_loss_epoch)

            if np.isnan(train_loss_epoch):
                self.logger.info("##### Training stopped: NaN value encountered in the training loss #####")
                break

            # Write current learning rate value:
            metric_dict['learning_rate'].append(utils.get_lr(optimizer))

            # #########################
            # ### Validation phase: ###
            # #########################

            self.model_to_train.eval()   # Set the model to evaluation model

            val_loss_batch = 0.
            # Iterate over the validation batches:
            for _, sample in enumerate(self.dataset_loaders['val']):
                # Handles GPU support
                inputs = sample['image'].to(self.train_device)
                ground_truth = sample['zernike'].to(self.train_device)

                # 1. Making prediction:
                estimation = self.model_to_train(inputs)
                # 2. Computing the loss for the current batch:
                loss = loss_function(torch.squeeze(estimation), torch.squeeze(ground_truth))

                val_loss_batch += 1 * loss.item() * inputs.size(0)

            # The loss over one epoch is the mean of the losses over the batches:
            val_loss_epoch = val_loss_batch / self.dataset_size["val"]
            self.logger.info('[{0}/{1}] val loss: {2:.6f}'.format(ep + 1, n_epochs, val_loss_epoch))

            # Update metrics:
            metric_dict['val_loss'].append(val_loss_epoch)

            # Use TensorBoard
            if useTensorBoard:
                self._tb_writer.add_scalars('loss',
                                           {'training_loss': train_loss_epoch, 
                                            'validation_loss': val_loss_epoch},
                                           ep)
                self._tb_writer.add_scalar('learning_rate', utils.get_lr(optimizer), ep)
                # self._tb_writer.add_scalar('validation_loss',
                #                            val_loss_epoch,
                #                            ep)

            # If scheduler is ReduceLROnPlateau we need to give current validation loss:
            optim_scheduler.step(metric_dict['val_loss'][ep])

            # Update current number of epochs:
            metric_dict["n_epochs"] = ep + 1
            # Save metrics:
            with open(metrics_path, 'w') as f:
                json.dump(metric_dict, f, indent=4)

            # Duration of whole epoch (train + validation):
            ep_min, ep_sec = divmod(time.time() - epoch_time, 60)
            self.logger.info('[{0}/{1}] duration: {2:.0f}m {3:.0f}s'.format(ep + 1, n_epochs, ep_min, ep_sec)) 

            # Check for early stopping:
            # 3 possibilities:
            # - if current val loss is the best: the weights are saved and training continues.
            # - if no improvement in validation loss since given number of epochs: training is stopped.
            # - if none of the above: nothing happens and training continues.
            early_stopping(val_loss_epoch, self.model_to_train)
            if early_stopping.early_stop is True:
                self.logger.info("=== Early stopping: the validation loss has not sufficiently decreased the last {} epochs ===".format(early_stopping.patience))
                break  # break the for loop regarding epochs

        tot_day, tot_hour, tot_min, tot_sec = utils.get_time(time.time() - since)
        self.logger.info('[-----] All epochs completed in {0:.0f}d {1:.0f}h {2:.0f}m {3:.0f}s'.format(tot_day, tot_hour, tot_min, tot_sec))

        if useTensorBoard:
            self._tb_writer.close()

        # Displaying the loss-epoch curve:
        if showTrainingCurve:
            self.plotLoss(metric_dict["train_loss"], metric_dict["val_loss"], save_figure=True) 

        return early_stopping.best_loss


    def generateDataLoaders(self, dataset_pp):
        # Splitting the dataset into training and validation:
        self.split = [self.config["split_train"] / 100, round(1 - (self.config["split_train"] / 100), 2)]
        self.dataset_loaders = {}
        self.dataset_loaders['train'], self.dataset_loaders['val'] = datapp.splitDataLoader(dataset = dataset_pp,
                                                                                conf=self.config,
                                                                                split=self.split,
                                                                                device=self.train_device)

        # Defining the training and validation data sizes:
        self.dataset_size = {
            'train': int(self.split[0] * float(self.data_info["nb_samples"])),
            'val': int(self.split[1] * float(self.data_info["nb_samples"]))
        }

    def dumpDataAttrs(self):
        # Saving the dataset information in json file:
        if not os.path.exists(self.config["model_dir"]):
            os.makedirs(self.config["model_dir"])
        datainfo_path = os.path.join(self.config["model_dir"], 'data_info.json')

        # Saving the data information:
        with open(datainfo_path, 'w') as f:
            # NpEncoder converts elements to the proper format:
            json.dump(self.data_info, f, indent=4, cls=utils.NpEncoder)


    def setConfig(self, conf_file, training_data_fname=None, **kwargs):
        if conf_file is None:
            conf_file = os.path.dirname(__file__) + "/config/training_config.yml"
            self.config = rt.read_conf(conf_file=conf_file)
        else:
            self.config = conf_file
        
        if training_data_fname is not None:
            self.config["training_data_fname"] = training_data_fname

        # setting kwargs
        for key, value in kwargs.items():
            if key in self.config.keys():
                self.config[key] = value
            else:
                #print(f"\n{fore.}WARNING{fmt.ENDC}: {str(key)} is not a valid key. Continuing with default.")
                self.logger.warn("{str(key)} is not a valid key. Continuing with default.")


        # print(f"\n{fmt.WARNING}Configs for training:{fmt.ENDC}")
        self.logger.info("Configs for training:")
        for key, val in self.config.items():
            print(f"{key}: {val}")

        #return fmt.inpQuit()

    def plotLoss(self, train_loss, val_loss, save_figure=False):
        # Plotting the training and validation loss:
        # fig, ax = plt.subplots()
        print(' toto plotting ')
        plt.clf()
        plt.plot(train_loss, label='Training loss')
        plt.plot(val_loss, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, train_loss[0]*1.5)
        # plt.set_xlabel('Epoch')
        # plt.set_ylabel('Loss')
        # plt.set_ylim(0, train_loss[0]*1.5)
        plt.legend()
        if save_figure:
            plt.savefig(self.config["model_dir"] + '/loss.png')
        # plt.show()
        plt.draw()
        plt.pause(0.01)