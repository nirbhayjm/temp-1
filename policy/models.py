import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Categorical, DiagGaussian
# from torch.distributions.categorical import Categorical
# from autoencoder.models import SimpleEncoder as Encoder
# from autoencoder.models import Encoder as Encoder
import utilities.utilities as utils
from utilities.utilities import init, init_normc_

from policy.base_models.clevr_base import FlattenMLPBase, CNNPlusMLPBase

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def flatten_last_two_dims(tensor):
    return tensor.view(*tensor.shape[:-2], tensor.shape[-1]*tensor.shape[-2])


class Policy(nn.Module):
    def __init__(self, observability, action_dims, base_model, base_kwargs=None):
        super(Policy, self).__init__()
        self.base_model = base_model
        if base_kwargs is None:
            base_kwargs = {}

        if observability == 'full':
            if self.base_model == 'mlp':
                self.base = FlattenMLPBase(**base_kwargs)
            elif self.base_model == 'cnn-mlp':
                self.base = CNNPlusMLPBase(**base_kwargs)
            else:
                raise ValueError
        else:
            raise NotImplementedError

        num_outputs = action_dims
        self.dist = Categorical(self.base.output_size, num_outputs)


    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        # assert torch.isnan(actor_features).sum().item() == 0
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        # if action_log_probs.mean() < -20:
        #     print("WHAT?!")
        #     import pdb; pdb.set_trace()
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action,
        get_entropy=True):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        if get_entropy:
            dist_entropy = dist.entropy().mean()
        else:
            dist_entropy = None

        return value, action_log_probs, dist_entropy, rnn_hxs, dist
