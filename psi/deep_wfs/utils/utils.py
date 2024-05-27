
import logging
import json
import os
import matplotlib.pyplot as plt

def plot_learningcurve(metrics, save=True, show=True, name='lrcurve.pdf', 
                       xlim=[None,None], ylim=[None,None], zernike=False):
    import numpy as np
    plt.figure()
    #x = np.arange(200)
    #plt.plot(x, np.array(metrics['train_loss' if not zernike else 'zernike_train_loss'])[x]/(0.8*np.log(x)), label='Training loss', color='blue')
    plt.plot(metrics['train_loss' if not zernike else 'zernike_train_loss'][:], label='Training loss', color='blue')
    plt.plot(metrics['val_loss' if not zernike else 'zernike_val_loss'][:], label='Validation loss', color='red')
    plt.legend()
    plt.grid()
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    if save: plt.savefig(name)
    if show: plt.show()
   

def get_metrics(model_dir=''):

    metrics_path = os.path.join(model_dir, 'metrics.json')

    with open(metrics_path) as f:
        metrics = json.load(f)
        return metrics

    # return None
