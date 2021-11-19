import torch


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def min_loss(x, t=2):
    pdist = torch.cdist(x, x, p=t)
    pdist = pdist + torch.diag(torch.tensor([1e10] * len(pdist), device=x.device))
    loss = -pdist.min(dim=-1).values.mean() #. pow(2).mul(-t).exp().mean().log()
    return loss
    # return -torch.pdist(y, p=2).min(dim=-1).values.mean()

def sort_2_loss(x, t=1):
    pdist = torch.cdist(x, x, p=2)
    pdist = pdist + torch.diag(torch.tensor([1e10] * len(pdist), device=x.device))
    sorted, indices = torch.sort(pdist, dim=-1) #
    loss = -sorted[:,t].mean() # take not the first but the seoncd
    return loss

def pairwise_loss(x, t=2):
    return -torch.pdist(x, p=t).mean()

__all__ = ['align_loss', 'uniform_loss', 'min_loss', 'pairwise_loss', "sort_2_loss"]
