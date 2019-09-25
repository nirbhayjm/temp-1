# Model defs for navigation

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as ds
from distributions import Categorical, DiagGaussian
from policy.base_models.clevr_base import AttributeEmbedding, NNBase, Flatten
from autoencoder.models import Encoder as Encoder
import utilities.utilities as utils
from utilities.utilities import init, init_normc_

def flatten_last_two_dims(tensor):
    return tensor.view(*tensor.shape[:-2], tensor.shape[-1]*tensor.shape[-2])


class GridCNNBase(NNBase):
    '''
    Module for encoding state grid into actor features
    (vector) and critic's state value prediction (scalar).
    '''
    def __init__(self,
                 input_channels,
                 omega_option_dims,
                 input_attr_dims,
                 recurrent=False,
                 hidden_size=512,
                 pretrained_encoder=False,
                 agent_cfg_dims=None):
        super().__init__(recurrent, hidden_size, hidden_size)

        self.input_channels = input_channels
        self.pretrained_encoder = pretrained_encoder
        self.agent_cfg_dims = agent_cfg_dims
        self.input_attr_dims = input_attr_dims
        self.hidden_size = hidden_size

        self.omega_option_dims = omega_option_dims

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2))


        self.omega_fc_actor = nn.Sequential(
            init_(nn.Linear(omega_option_dims, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.omega_fc_critic = nn.Sequential(
            init_(nn.Linear(omega_option_dims, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )


        encoder_dim = (2*2 + 7*7 + 15*15)*3
        # base_feat_dim = encoder_dim + embed_size


        if pretrained_encoder:
            self.encoder = Encoder()
            self.after_encoder = nn.Sequential(
                init_(nn.Linear(encoder_dim, hidden_size)),
                nn.ReLU(),
            )
            self.triplet_fc = nn.Sequential(
                init_(nn.Linear(3 * hidden_size, hidden_size)),
                nn.ReLU(),
            )
        else:
            init_ = lambda m: init(m,
                nn.init.orthogonal_,
                lambda x: nn.init.constant_(x, 0),
                nn.init.calculate_gain('relu'))

            self.encoder = nn.Sequential(
                # 30 x 30
                init_(nn.Conv2d(input_channels, 10, 1, stride=1)),
                # nn.MaxPool2d(2, 2),
                nn.ReLU(),
                # 30 x 30
                init_(nn.Conv2d(10, 32, 4, stride=2, padding=1)),
                nn.ReLU(),
                # 15 x 15
                init_(nn.Conv2d(32, 32, 3, stride=2, padding=1)),
                # 8 x 8
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(32 * 8 * 8, hidden_size)),
                nn.ReLU(),
            )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        if agent_cfg_dims is not None:
            fc_input_size = hidden_size + agent_cfg_dims
        else:
            fc_input_size = hidden_size

        self.actor_fc = nn.Sequential(
                init_(nn.Linear(fc_input_size + hidden_size, hidden_size)),
                nn.ReLU(),
                init_(nn.Linear(hidden_size, hidden_size)),
        )

        self.critic_linear = nn.Sequential(
            init_(nn.Linear(fc_input_size + hidden_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, 1)),
        )


        #self.target_embed = AttributeEmbedding(
        #    embed_type='k-hot',
        #    input_dim=self.input_attr_dims,
        #    embed_size=self.hidden_size,
        #    hidden_size=self.hidden_size,
        #    output_size=self.hidden_size)

        self.train()

    def init_pretrained_encoder(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _encode_img(self, input_image):
        if self.pretrained_encoder:
            feats = []
            for img_ in torch.split(input_image, 3, dim=1):
                s1, s2, s3 = self.encoder(img_)
                s1 = flatten_last_two_dims(s1)
                s2 = flatten_last_two_dims(s2)
                s3 = flatten_last_two_dims(s3)
                img_feat = torch.cat([s1, s2, s3], 2)
                img_feat = img_feat.view(img_feat.shape[0], -1)
                feats.append(self.after_encoder(img_feat))
            return self.triplet_fc(torch.cat(feats, 1))
        else:
            return self.encoder(input_image)

    def forward(self, observation, rnn_hxs, masks, tuple_obs=False):
        omega_option, maps, agent_info = observation

        encoded_map = self._encode_img(maps)

        # [NOTE] : Check if we need to encode 'self.omega_option' again.
        omega_option_actor = self.omega_fc_actor(omega_option)
        # Gradients for critic should not pass through to omega_option
        omega_option_critic = self.omega_fc_critic(omega_option.detach())

        actor_feat = torch.cat([omega_option_actor, encoded_map, agent_info], 1)
        critic_feat = torch.cat([omega_option_critic, encoded_map, agent_info], 1)

        actor_feat = self.actor_fc(actor_feat)

        if self.is_recurrent:
            actor_feat, rnn_hxs = self._forward_gru(
                actor_feat, rnn_hxs, masks)

        return self.critic_linear(critic_feat), actor_feat, rnn_hxs
