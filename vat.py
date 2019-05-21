import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_div_with_logit(q_logit: torch.Tensor, p_logit: torch.Tensor):
    '''
    :param q_logit:it is like the y in the ce loss
    :param p_logit: it is the logit to be proched to q_logit
    :return:
    '''
    assert not q_logit.requires_grad
    assert p_logit.requires_grad
    q = F.softmax(q_logit, dim=1)
    p = F.softmax(p_logit, dim=1)
    logq = torch.log(q + 1e-8)
    logp = torch.log(p + 1e-8)

    qlogq = (q * logq).sum(dim=1)
    qlogp = (q * logp).sum(dim=1)
    return qlogq - qlogp


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True)+1e-16
    # assert torch.allclose(d.view(d.shape[0], -1).norm(dim=1),
    #                       torch.ones_like(d.view(d.shape[0], -1).norm(dim=1))), f'failed in d2 normlization,\
    #                        {d.view(d.shape[0], -1).norm(dim=1)}'
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = model(x)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                adv_distance = kl_div_with_logit(pred, pred_hat).mean()
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.eps
            pred_hat = model((x + r_adv).detach())
            lds = kl_div_with_logit(pred, pred_hat).mean()

        return lds


def _entropy(logits):
    p = F.softmax(logits, dim=1)
    return -torch.mean(torch.sum(p * F.log_softmax(logits, dim=1), dim=1))


class VATGenerator(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATGenerator, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = model(x)
        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                adv_distance = kl_div_with_logit(pred, pred_hat).mean()
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            adv_img = x + r_adv
            # pred_hat = model(x + r_adv)
            # logp_hat = F.log_softmax(pred_hat, dim=1)
            # lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return adv_img.detach(), pred
