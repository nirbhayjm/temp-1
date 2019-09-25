from typing import Dict, Optional, Tuple

import math
import collections
import operator
import numpy as np
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import spaces
from distributions import Categorical, DiagGaussian
# from torch.distributions.categorical import Categorical
# from autoencoder.models import SimpleEncoder as Encoder
# from autoencoder.models import Encoder as Encoder
import utilities.utilities as utils
from utilities.utilities import init, init_normc_

def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def flatten_last_two_dims(tensor):
    return tensor.view(*tensor.shape[:-2], tensor.shape[-1]*tensor.shape[-2])


class NNBase(nn.Module):
    '''
    NNBase, unmodified from github.com/ikostrikov/pytorch-a2c-ppo-acktr
    '''
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())


            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]


            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self,
                 input_channels,
                 target_dim,
                 target_embed_type,
                 # num_attributes,
                 embed_size,
                 recurrent=False,
                 hidden_size=512,
                 input_size=64,
                 pretrained_encoder=False,
                 agent_cfg_dims=None):
        super().__init__(recurrent, hidden_size, hidden_size)

        self.input_size = input_size
        self.target_dim = target_dim
        self.embed_size = embed_size
        self.input_channels = input_channels
        self.pretrained_encoder = pretrained_encoder
        self.agent_cfg_dims = agent_cfg_dims

        # self.num_attributes = num_attributes
        self.target_embed_type = target_embed_type

        self.target_embed_actor = AttributeEmbedding(
            embed_type=target_embed_type,
            input_dim=target_dim,
            embed_size=embed_size,
            hidden_size=hidden_size,
            output_size=hidden_size)

        self.target_embed_critic = AttributeEmbedding(
            embed_type=target_embed_type,
            input_dim=target_dim,
            embed_size=embed_size,
            hidden_size=hidden_size,
            output_size=hidden_size)

        self.init_embedding_weights()

        encoder_dim = (2*2 + 7*7 + 15*15)*3
        # base_feat_dim = encoder_dim + embed_size

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        if pretrained_encoder:
            self.encoder = Encoder()
            self.after_encoder = nn.Sequential(
                nn.Linear(encoder_dim, hidden_size),
                nn.ReLU(),
            )
            self.triplet_fc = nn.Sequential(
                nn.Linear(3 * hidden_size, hidden_size),
                nn.ReLU(),
            )
        else:
            # [NOTE] : If we switch to some other gridworld, this has to be
            # taken care of.

            if self.input_size == 30:
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
            else:
                raise ValueError

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        if agent_cfg_dims is not None:
            fc_input_size = hidden_size + agent_cfg_dims
        else:
            fc_input_size = hidden_size

        if target_embed_type == 'one-hot':
            self.actor_fc = nn.Sequential(
                nn.Linear(fc_input_size + embed_size, hidden_size),
                nn.ReLU(),
            )
            self.critic_linear = init_(
                nn.Linear(hidden_size + embed_size, 1))

        elif target_embed_type == 'k-hot':
            self.actor_fc = nn.Sequential(
                nn.Linear(fc_input_size + hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.critic_linear = nn.Sequential(
                init_(nn.Linear(fc_input_size + hidden_size, hidden_size)),
                nn.ReLU(),
                init_(nn.Linear(hidden_size, 1)),
            )

        self.train()


    def init_embedding_weights(self):
        for module in self.modules():
            # Initialize weights
            name = type(module).__name__
            # If linear or embedding
            if name in ['Embedding']:
                fanIn = module.weight.data.size(0)
                fanOut = module.weight.data.size(1)

                factor = math.sqrt(2.0 / (fanIn + fanOut))
                weight = torch.randn(fanIn, fanOut) * factor
                module.weight.data.copy_(weight)

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
        SZ = self.input_size
        if tuple_obs:
            if self.agent_cfg_dims is not None:
                target, inputs, agent_info = observation
            else:
                target, inputs = observation
            target = target.long()

        elif self.target_embed_type == 'one-hot':
            target = observation[:, :1].long()[:, 0]
            inputs = observation[:, 1:].view(
                observation.size(0), self.input_channels, SZ, SZ)

        elif self.target_embed_type == 'k-hot':
            target = observation[:, :len(self.target_dim)].long()
            inputs = observation[:, len(self.target_dim):].view(
                observation.size(0), self.input_channels, SZ, SZ)

        target_actor = self.target_embed_actor(target)
        target_critic = self.target_embed_critic(target)

        # x = self.encoder(inputs / 255.0)
        # assert inputs.cpu().numpy().max() < 1 + 1e-6
        x = self._encode_img(inputs)

        if self.agent_cfg_dims is not None:
            x_actor = torch.cat([x, target_actor, agent_info], 1)
            x_critic = torch.cat([x, target_critic, agent_info], 1)
        else:
            x_actor = torch.cat([x, target_actor], 1)
            x_critic = torch.cat([x, target_critic], 1)

        x_actor = self.actor_fc(x_actor)

        if self.is_recurrent:
            x_actor, rnn_hxs = self._forward_gru(
                x_actor, rnn_hxs, masks)

        return self.critic_linear(x_critic), x_actor, rnn_hxs


class FullObsBase(CNNBase):
        def __init__(self,
                     input_channels,
                     target_dim,
                     target_embed_type,
                     embed_size,
                     obs_embed_size,
                     agent_cfg_dims,
                     recurrent=False,
                     hidden_size=512):
            super().__init__(input_channels,
                             target_dim,
                             target_embed_type,
                             embed_size,
                             recurrent,
                             hidden_size,
                             input_size=30,
                             pretrained_encoder=False,
                             agent_cfg_dims=agent_cfg_dims)

            self.AGENT_CFG_DIMS = agent_cfg_dims
            self.obs_embed_size = obs_embed_size
            self.target_embed_type = target_embed_type

            if self.target_embed_type == 'one-hot':
                pass

            elif self.target_embed_type == 'k-hot':
                self.obs_embedding = AttributeEmbedding(
                    embed_type=target_embed_type,
                    input_dim=target_dim,
                    embed_size=self.obs_embed_size,
                    hidden_size=self.obs_embed_size,
                    output_size=self.obs_embed_size)

        def forward(self, observation, rnn_hxs, masks):
            if self.target_embed_type == 'one-hot':
                obs_embed = observation
            elif self.target_embed_type == 'k-hot':
                obs = observation
                target = obs[:, :len(self.target_dim)].float()
                # res_ = obs[:, len(self.target_dim):] #.long()
                res_ = obs[:, len(self.target_dim):].float()
                agent_info = res_[:, :self.AGENT_CFG_DIMS]
                maps_flat = res_[:, self.AGENT_CFG_DIMS:]

                # maps = maps_flat.view(maps_flat.shape[0],
                #     self.AGENT_CFG_DIMS + sum(self.target_dim), 30, 30)
                maps = maps_flat.view(maps_flat.shape[0],
                    sum(self.target_dim), 30, 30)
                maps = maps.float()
                # agent_map = maps[:, :1].float()
                # cmap = maps[:, 1:]
                # obs_embed = self.obs_embedding(cmap)
                # obs_embed = obs_embed.permute(0, 3, 1, 2)
                # obs_embed = torch.cat([agent_map, obs_embed], 1)

                obs_embed = maps
                # obs_embed = (target, obs_embed)
                obs_embed = (target, obs_embed, agent_info)

            value, x_actor, rnn_hxs = super().forward(
                obs_embed, rnn_hxs, masks, tuple_obs=True)
            return value, x_actor, rnn_hxs


class MLPBase(NNBase):
    '''
    MLPBase, unmodified from github.com/ikostrikov/pytorch-a2c-ppo-acktr
    '''
    def __init__(
        self,
        num_inputs: int,
        recurrent: bool = False,
        hidden_size: int = 64,
        use_critic: bool = True,
        critic_detach: bool = True,
    ):
        super().__init__(recurrent, num_inputs, hidden_size)
        self.critic_detach = critic_detach

        self.use_critic = use_critic
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            # nn.LeakyReLU(0.1),
            # nn.ELU(),
            nn.Tanh(),
            # init_(nn.Linear(hidden_size, hidden_size)),
            # nn.LeakyReLU(0.1),
            # nn.ELU(),
            # init_(nn.Linear(hidden_size, hidden_size)),
            # nn.LeakyReLU(0.1),
        )

        if use_critic:
            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)),
                # nn.LeakyReLU(0.1),
                # nn.ELU(),
                nn.Tanh(),
                # init_(nn.Linear(hidden_size, hidden_size)),
                # # nn.LeakyReLU(0.1),
                # nn.ELU(),
                # init_(nn.Linear(hidden_size, hidden_size)),
                # nn.LeakyReLU(0.1),
            )
            self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs=None, masks=None):
        x = inputs

        if self.is_recurrent:
            assert rnn_hxs is not None
            assert masks is not None
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_actor = self.actor(x)

        if self.use_critic:
            if self.critic_detach:
                hidden_critic = self.critic(x.detach())
            else:
                hidden_critic = self.critic(x)
            return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
        else:
            return hidden_actor, rnn_hxs


class FlattenMLPBase(MLPBase):
    def __init__(
        self,
        obs_spaces: collections.OrderedDict,
        recurrent: bool = False,
        hidden_size: int = 64,
        use_critic: bool = True,
        critic_detach: bool = True,
    ):
        num_inputs = 0
        self.obs_keys = obs_spaces.keys()

        for key, space in obs_spaces.items():
            flat_size = reduce(operator.mul, space.shape, 1)
            num_inputs += flat_size

        super().__init__(
            num_inputs=num_inputs,
            recurrent=recurrent,
            hidden_size=hidden_size,
            use_critic=use_critic,
            critic_detach=critic_detach,
        )

    def forward(self, inputs, rnn_hxs=None, masks=None):
        flat_inputs = []

        for key in self.obs_keys:
            inp = inputs[key]
            inp_flat = inp.view(inp.size(0), -1)
            flat_inputs.append(inp_flat)

        new_input = torch.cat(flat_inputs, 1)
        return super().forward(new_input, rnn_hxs, masks)


class CNNPlusMLPBase(FlattenMLPBase):
    def __init__(
        self,
        obs_spaces: collections.OrderedDict,
        recurrent: bool = False,
        hidden_size: int = 64,
        use_critic: bool = True,
        critic_detach: bool = True,
    ):
        num_inputs = 0
        self.obs_keys = obs_spaces.keys()
        self.image_space = obs_spaces['image']
        mlp_obs_spaces = obs_spaces.copy()
        mlp_obs_spaces.update({
            'image': spaces.Box(
                low=0.0, # Arbitrary value
                high=1.0, # Arbitrary value
                shape=(hidden_size,),
                dtype='float',
            )
        })
        self.mlp_obs_keys = mlp_obs_spaces.keys()

        super().__init__(
            obs_spaces=mlp_obs_spaces,
            recurrent=recurrent,
            hidden_size=hidden_size,
            use_critic=use_critic,
            critic_detach=critic_detach,
        )

        _neg_slope = 0.1
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('leaky_relu', param=_neg_slope))

        NEW_CNN = True

        H, W, num_channels = self.image_space.shape
        if NEW_CNN:
            self.cnn = nn.Sequential(
                init_(nn.Conv2d(num_channels, 16, (3,3), padding=1)),
                nn.ReLU(),
                # nn.MaxPool2d((2, 2)),
                init_(nn.Conv2d(16, 32, (2,2))),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, (2,2))),
                nn.ReLU(),
                Flatten(),
            )
        else:
            self.cnn = nn.Sequential(
                init_(nn.Conv2d(num_channels, 16, 1, stride=1)),
                # nn.LeakyReLU(_neg_slope),
                nn.ELU(),
                init_(nn.Conv2d(16, 8, 3, stride=1, padding=2)),
                # nn.LeakyReLU(_neg_slope),
                nn.ELU(),
                # init_(nn.Conv2d(64, 64, 5, stride=1, padding=2)),
                # nn.LeakyReLU(_neg_slope),
                Flatten(),
            )
        output_h_w, out_channels = utils.conv_sequential_output_shape((H, W), self.cnn)
        h_w_prod = output_h_w[0] * output_h_w[1]
        self.fc = nn.Sequential(
            init_(nn.Linear(out_channels * h_w_prod, hidden_size)),
            # nn.LeakyReLU(_neg_slope),
            # nn.ELU(),
        )
        self.apply(initialize_parameters)

    def cnn_forward(self, image):
        # NOTE: The image.clone() here is important as self.cnn
        # modifies the image in-place by normalizing it.
        # This affects the downstream image stored in observations.
        cnn_feats = self.cnn(image.clone().permute([0, 3, 1, 2]))
        output_feats = self.fc(cnn_feats)
        return output_feats

    def forward(self, inputs, rnn_hxs=None, masks=None):
        image_feats = self.cnn_forward(inputs['image'])
        new_inputs = {key: value for key, value in inputs.items() if key != 'image'}
        new_inputs['image'] = image_feats
        return super().forward(inputs=new_inputs, rnn_hxs=rnn_hxs, masks=masks)


class AttributeEmbedding(nn.Module):
    '''
    Module for embedding attributes. If there are a total of
    K attributes where the i-th attribute can take n_i values,
    i \in {0, 1, ..., K-1}, then two embedding types are supported:

    "one-hot":
        All possible value assignments for K attributes i.e.
        N = (n_0 * n_1 * ... * n_K-1) are each mapped to a D
        dimensional embedding using an (N x D) embedding table.

    "k-hot":
        K embedding tables are created, one for each attribute
        and of size (n_i x H), i \in {0, 1, ..., K-1}. After
        embedding all K attributes to their respective H dimensional
        embeddings, they are concatenated and passed through
        an MLP to get a fixed dimensional embedding of size output_size

    Arguments:
        embed_type: "one-hot" or "k-hot"

        input_dim: Tuple specifying (n_0, n_1, ... n_K-1)

        embed_size: Final embedding size for "one-hot"

        hidden_size: Overloaded parameter specfying hidden size of
            MLPs and also intermediate embedding size of "k-hot"

        output_size: Final embedding size for "k-hot", after
            concatenating K embeddings through an MLP

    '''
    def __init__(self,
                 embed_type: str,
                 input_attr_dims: Tuple[int],
                 embed_size: int,
                 hidden_size: int, #TODO: Stop overloading hidden_size
                 # num_attributes,
                 output_size: Optional[int] = None):
        super().__init__()

        self.embed_type = embed_type
        self.embed_size = embed_size
        self.input_attr_dims = input_attr_dims
        # self.num_attributes = num_attributes

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2))

        if embed_type == 'one-hot':
            self.main_embed = nn.Embedding(
                input_attr_dims, embed_size)

        elif embed_type == 'k-hot':
            assert len(input_attr_dims) > 1, \
                "Use one-hot for single attribute"
            assert output_size != None, \
                "Output size needed for k-hot embeddings"

            self.main_embed = nn.ModuleList([
                nn.Sequential(
                    nn.Embedding(dim, embed_size),
                    init_(nn.Linear(embed_size, hidden_size)),
                    nn.LeakyReLU(0.1),
                    init_(nn.Linear(hidden_size, hidden_size)),
                ) for dim in input_attr_dims])

            self.fc = nn.Sequential(
                    init_(nn.Linear(len(input_attr_dims) * hidden_size, \
                        hidden_size)),
                nn.LeakyReLU(0.1),
                init_(nn.Linear(hidden_size, output_size)),
            )
        self._init_embedding_tables()

    def _init_embedding_tables(self):
        for module in self.modules():
            name = type(module).__name__
            if name in ['Embedding']:
                fanIn = module.weight.data.size(0)
                fanOut = module.weight.data.size(1)

                factor = math.sqrt(2.0 / (fanIn + fanOut))
                weight = torch.randn(fanIn, fanOut) * factor
                module.weight.data.copy_(weight)

    def forward(self, target):
        if self.embed_type == 'one-hot':
            target = self.main_embed(target)

        elif self.embed_type == 'k-hot':
            target = [
                self.main_embed[attr_idx](target[:, attr_idx]) \
                for attr_idx in range(len(self.input_attr_dims))]

            target = self.fc(torch.cat(target,-1))

        return target
