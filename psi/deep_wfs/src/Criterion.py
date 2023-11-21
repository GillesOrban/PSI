# used to define the loss used

import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss(reduction='none')

    def forward(self, yhat, y):
        # Compute the sqrt() only on the Zernike coefficients and not on the batch elements:
        try:
            loss = torch.mean(torch.sqrt(torch.sum(self.mse(yhat, y), dim=1)))
        except IndexError:
            # In the case where there is only one batch element:
            loss = torch.sqrt(torch.sum(self.mse(yhat, y)))
        return loss
