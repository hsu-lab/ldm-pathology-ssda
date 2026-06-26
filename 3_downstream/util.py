import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logp = F.log_softmax(inputs, dim=1)
        p = torch.exp(logp)
        loss = F.nll_loss((1 - p) ** self.gamma * logp, targets, reduction='none')
        if isinstance(self.alpha, (list, torch.Tensor)):
            alpha_t = self.alpha[targets]
            loss *= alpha_t.to(inputs.device)
        elif self.alpha != 1.0:
            loss *= self.alpha
        return loss.mean() if self.reduction == 'mean' else loss.sum()
