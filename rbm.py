import torch
from torch import nn
from torch.nn import functional as F


class RBM(nn.Module):

    def __init__(self):
        super(RBM, self).__init__()

        self.W = nn.Parameter(
            torch.randn(784, 128),
            requires_grad=True
        )

        self.b = nn.Parameter(
            torch.randn(1, 784),
            requires_grad=True
        )

        self.c = nn.Parameter(
            torch.randn(1, 128),
            requires_grad=True
        )

    def forward(self, v0, k=1):
        v = v0

        for i in range(k):
            h, p_h = self._visible_to_hidden(v)
            v, p_v = self._hidden_to_visible(h)

        return v, p_v

    def _visible_to_hidden(self, v):
        logit = torch.mm(v, self.W) + self.c
        prob = torch.sigmoid(logit)
        h = torch.distributions.Bernoulli(probs=prob).sample()
        return h, prob

    def _hidden_to_visible(self, h):
        logit = torch.mm(h, self.W.t()) + self.b
        prob = torch.sigmoid(logit)
        v = torch.distributions.Bernoulli(probs=prob).sample()
        return v, prob

    def free_energy(self, v):
        logit = torch.mm(v, self.W) + self.c
        # fe = -torch.mm(v, self.b.t()) - torch.sum(torch.log(1 + torch.exp(logit)), dim=1)
        fe = -torch.mm(v, self.b.t()) - torch.sum(F.softplus(logit), dim=1)

        return fe
