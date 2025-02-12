'''
** Context (from SAM github, see below): **
@hjq133: The suggested usage can potentially cause problems if you use batch normalization. 
The running statistics are computed in both forward passes, 
but they should be computed only for the first one. 
A possible solution is to set BN momentum to zero (kindly suggested by @ahmdtaha) 
to bypass the running statistics during the second pass. 
An example usage is on lines 51 and 58 in example/train.py:

source:
https://github.com/davda54/sam/blob/main/example/utility/bypass_bn.py
'''

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)