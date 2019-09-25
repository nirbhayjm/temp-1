# Model defs for navigation

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as ds
from distributions import Categorical, DiagGaussian
from policy.base_models.clevr_base import AttributeEmbedding, NNBase, Flatten
# from autoencoder.models import Encoder as Encoder
import utilities.utilities as utils
from utilities.utilities import init, init_normc_

def flatten_last_two_dims(tensor):
    return tensor.view(*tensor.shape[:-2], tensor.shape[-1]*tensor.shape[-2])


class CNNBase(NNBase):
    '''
    CNNBase module adapted from ikostrikov/pytorch-a2c-ppo-acktr

    Module for encoding state observations into actor features
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


class CNNSimpleBase(NNBase):
    '''
    Module for encoding state observations into actor features
    (vector) and critic's state value prediction (scalar).
    '''
    def __init__(self,
                 input_channels,
                 goal_output_size,
                 goal_attr_dims,
                 state_encoder_hidden_size,
                 recurrent=False,
                 pretrained_encoder=False,
                 agent_cfg_dims=None):

        super().__init__(recurrent,
                state_encoder_hidden_size,
                state_encoder_hidden_size)

        # self.model = model
        self.input_channels = input_channels
        self.pretrained_encoder = pretrained_encoder
        self.agent_cfg_dims = agent_cfg_dims
        self.goal_attr_dims = goal_attr_dims
        self.state_encoder_hidden_size = state_encoder_hidden_size

        self.goal_output_size = goal_output_size

        self.goal_fc_actor = nn.Sequential(
            nn.Linear(goal_output_size, goal_output_size),
            nn.ReLU(),
        )

        self.goal_fc_critic = nn.Sequential(
            nn.Linear(goal_output_size, goal_output_size),
            nn.ReLU(),
        )


        encoder_dim = (2*2 + 7*7 + 15*15)*3
        # base_feat_dim = encoder_dim + embed_size

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.map_encoder = nn.Sequential(
            # 30 x 30
            init_(nn.Conv2d(input_channels, 10, 1, stride=1)),
            # nn.MaxPool2d(2, 2),
            nn.ReLU(),
            # 30 x 30
            init_(nn.Conv2d(10, 10, 4, stride=2, padding=1)),
            nn.ReLU(),
            # 15 x 15
            init_(nn.Conv2d(10, 10, 3, stride=2, padding=1)),
            ## 8 x 8
            #nn.ReLU(),
            Flatten(),
            init_(nn.Linear(10 * 8 * 8, state_encoder_hidden_size)),
            nn.ReLU(),
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        if agent_cfg_dims is not None:
            fc_input_size = self.state_encoder_hidden_size + \
                             self.agent_cfg_dims + \
                             self.goal_output_size
        else:
            fc_input_size = hidden_size

        self.actor_fc = nn.Sequential(
                nn.Linear(fc_input_size, state_encoder_hidden_size),
                nn.ReLU(),
                nn.Linear(state_encoder_hidden_size, state_encoder_hidden_size),
        )

        self.critic_linear = nn.Sequential(
            init_(nn.Linear(fc_input_size, state_encoder_hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(state_encoder_hidden_size, 1)),
        )

        self.train()

    def freeze_encoder(self):
        for param in self.map_encoder.parameters():
            param.requires_grad = False

    def forward(self, observation, rnn_hxs, masks, tuple_obs=False):
        goal, agent_info, maps = observation

        encoded_map = self.map_encoder(maps)

        goal_actor = self.goal_fc_actor(goal)
        goal_critic = self.goal_fc_critic(goal)

        actor_feat = torch.cat([goal_actor, encoded_map, agent_info], 1)
        critic_feat = torch.cat([goal_critic, encoded_map, agent_info], 1)

        actor_feat = self.actor_fc(actor_feat)

        if self.is_recurrent:
            actor_feat, rnn_hxs = self._forward_gru(
                actor_feat, rnn_hxs, masks)

        return self.critic_linear(critic_feat), actor_feat, rnn_hxs


class FullyObservedSimpleBase(CNNSimpleBase):

    def __init__(self,
                input_channels,
                agent_cfg_dims,
                goal_attr_dims,
                goal_output_size,
                recurrent=False,
                state_encoder_hidden_size=128):

        super().__init__(input_channels,
                         goal_output_size,
                         goal_attr_dims,
                         state_encoder_hidden_size,
                         recurrent,
                         pretrained_encoder=False,
                         agent_cfg_dims=agent_cfg_dims)
        """
        Attributes:
            'goal_attr_dims' : A tuple of number of values for each attribute
                                of the goal.

            'goal_output_size' : Size of the goal encoding. In case of conditional
                            model, this is the ize of the output of GoalEncoder.
                            For Hierarchical model, this is the size of the Omega
                            vector.

            'agent_cfg_dims' : Dimensions of the agent configuration. For now, it
                            is 2, representing the xy location of the agent on the
                            grid.
        """

        self.goal_attr_dims = goal_attr_dims
        self.goal_output_size = goal_output_size
        self.agent_cfg_dims = agent_cfg_dims

    def forward(self, observation, rnn_hxs, masks):
        obs = observation

        target = obs[:, :self.goal_output_size]

        obs = obs[:, self.goal_output_size: ]

        agent_info = obs[:, :self.agent_cfg_dims]

        obs = obs[:, self.agent_cfg_dims: ]

        maps_flat = obs

        maps = maps_flat.view(maps_flat.shape[0],
            sum(self.goal_attr_dims), 30, 30).float()

        obs = (target, agent_info, maps)

        value, x_actor, rnn_hxs = super().forward(
            obs, rnn_hxs, masks)

        return value, x_actor, rnn_hxs


class FullyObservedBase(CNNBase):
    '''
    Module for encoding states in the fully observed (MDP)
    setting.

    It is a wrapper over CNNBase as it just does some
    preprocessing of the environment observation and agent
    configuration to bring it to the proper input format
    for CNNBase. This wrapper was created because a standalone
    class for encoding fully observed states would have large
    overlap with CNNBase.
    '''
    def __init__(self,
                input_channels,
                input_attr_dims,
                agent_cfg_dims,
                omega_option_dims,
                recurrent=False,
                hidden_size=512):

        super().__init__(input_channels,
                         omega_option_dims,
                         input_attr_dims,
                         recurrent,
                         hidden_size,
                         pretrained_encoder=False,
                         agent_cfg_dims=agent_cfg_dims)

        self.AGENT_CFG_DIMS = agent_cfg_dims
        self.omega_option_dims = omega_option_dims
        self.input_attr_dims = input_attr_dims

    def forward(self, observation, rnn_hxs, masks):
        # B*(2+30*30*15) for 'full' observability
        obs = observation

        omega_option = obs[:, :self.omega_option_dims]

        obs = obs[:, self.omega_option_dims:]
        agent_info = obs[:, :self.AGENT_CFG_DIMS]

        obs = obs[:, self.AGENT_CFG_DIMS:]
        maps_flat = obs

        maps = maps_flat.view(maps_flat.shape[0],
            sum(self.input_attr_dims), 30, 30).float()

        obs = (omega_option, maps, agent_info)

        value, x_actor, rnn_hxs = super().forward(
            obs, rnn_hxs, masks, tuple_obs=True)
        return value, x_actor, rnn_hxs
