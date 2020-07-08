import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, bias=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight.half()))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1)).half()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class Circle(nn.Module):
    def __init__(self, in_features, out_features, s=128.0, m=0.25, bias=False):
        super(Circle, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, features, targets):
        sim_mat = F.linear(F.normalize(features), F.normalize(self.weight))
        alpha_p = F.relu(-sim_mat.detach() + 1 + self.m)
        alpha_n = F.relu(sim_mat.detach() + self.m)
        delta_p = 1 - self.m
        delta_n = self.m

        s_p = self.s * alpha_p * (sim_mat - delta_p)
        s_n = self.s * alpha_n * (sim_mat - delta_n)

        one_hot = torch.zeros(sim_mat.size(), device='cuda')
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)

        pred_class_logits = one_hot * s_p + (1.0 - one_hot) * s_n

        return pred_class_logits


class CrossEntropyLoss(object):
    """
    A class that stores information and compute losses about outputs of a Baseline head.
    """

    def __init__(self, num_classes, epsilon):
        self._num_classes = num_classes
        self._eps = epsilon
        self._alpha = 0.3

    def __call__(self, pred_class_logits, gt_classes):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        if self._eps >= 0:
            smooth_param = self._eps
        else:
            # adaptive lsr
            soft_label = F.softmax(pred_class_logits, dim=1)
            smooth_param = self._alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

        log_probs = F.log_softmax(pred_class_logits, dim=1)
        with torch.no_grad():
            targets = torch.ones_like(log_probs)
            targets *= smooth_param / (self._num_classes - 1)
            targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

        loss = (-targets * log_probs).mean(0).sum()
        return loss


def mixed_loss(score, logit, label_a, label_c):
    loss_a = F.cross_entropy(score, label_a)
    loss_c = F.cross_entropy(logit, label_c)
    return loss_a, loss_c