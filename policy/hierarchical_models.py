# Model defs for navigation

from typing import Dict, Optional, Tuple

import math
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as ds
from distributions import Categorical, DiagGaussian
# from autoencoder.models import Encoder as Encoder
import utilities.utilities as utils
from utilities.utilities import init, init_normc_

from policy.models import Policy
from policy.base_models.clevr_base import (AttributeEmbedding,
                                           Flatten,
                                           FlattenMLPBase,
                                           CNNPlusMLPBase)
from policy.base_models.hierarchical_base import FullyObservedBase

def flatten_last_two_dims(tensor):
    return tensor.view(*tensor.shape[:-2], tensor.shape[-1]*tensor.shape[-2])


class TrajectoryEncoder(nn.Module):
    '''
    Inference model to predict omega (option) given a trajectory.
    Trajectories are currently specified by their final states.
    '''

    def __init__(
        self,
        input_type: str,
        ic_mode: str,
        observability: str,
        option_space: str,
        omega_option_dims: int,
        hidden_size: int,
        base_model: str,
        base_kwargs: Dict,
    ):
        super().__init__()

        assert input_type in \
            ['final_state', 'final_and_initial_state']
        assert option_space in ['continuous', 'discrete']
        assert 'mission' not in base_kwargs['obs_spaces']
        assert ic_mode in ['vic', 'diyan', 'valor']

        if ic_mode != 'vic':
            input_type = 'final_state'

        self.input_type = input_type
        self.ic_mode = ic_mode
        self.hidden_size = hidden_size
        self.option_space = option_space
        self.omega_option_dims = omega_option_dims
        self.observability = observability
        self.base_model = base_model

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        if base_kwargs is None:
            base_kwargs = {}

        if ic_mode == 'valor':
            base_kwargs['recurrent'] = True

        if self.base_model == 'cnn-mlp' and \
        'image' not in base_kwargs['obs_spaces']:
            self.base_model = 'mlp'
            print("Switching to MLP for TrajectoryEncoder since no Image"
                  " present in obs_spaces!")

        assert observability == 'full'
        if self.base_model == 'mlp':
            self.base = FlattenMLPBase(**base_kwargs)
        elif self.base_model == 'cnn-mlp':
            self.base = CNNPlusMLPBase(**base_kwargs)
        else:
            raise ValueError

        state_feat_dim = base_kwargs['hidden_size']
        if input_type == 'final_and_initial_state':
            state_feat_dim *= 2

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2))

        self.fc = nn.Sequential(
            init_(nn.Linear(state_feat_dim, hidden_size)),
            # nn.LeakyReLU(0.1),
            nn.ELU(),
            # init_(nn.Linear(hidden_size, hidden_size)),
            # nn.LeakyReLU(0.1),
            # init_(nn.Linear(hidden_size, hidden_size)),
        )

        if self.option_space == 'continuous':
            self.fc12 = init_(nn.Linear(
                hidden_size, 2 * omega_option_dims))
        else:
            self.fc_logits = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)),
                # nn.LeakyReLU(0.1),
                nn.ELU(),
            )
            self.dist = Categorical(hidden_size, self.omega_option_dims)


    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def encode_state_sequence(self, trajectory, masks):
        assert masks is not None
        num_steps, num_processes = masks.shape[:2]

        outs_all = []
        hxs = masks.new_zeros(size=(num_processes,
            self.recurrent_hidden_state_size))

        for step in range(num_steps):
            out, hxs = self.base(
                inputs=trajectory[step],
                rnn_hxs=hxs,
                masks=masks[step],
            )
            outs_all.append(out)

        outs_cat = torch.stack(outs_all, 0)

        final_t = masks.sum(0).long() - 1
        final_t = final_t.expand_as(outs_cat)[:1]

        final_step_outs = outs_cat.gather(
            dim=0, index=final_t).squeeze(0)
        return final_step_outs

    def forward(self, trajectory, resizing_shape=None,
        masks=None):
        if self.ic_mode == 'valor':
            obs_feats = self.encode_state_sequence(
                trajectory=trajectory, masks=masks)

        elif self.input_type == 'final_state':
            final_state = trajectory
            obs_feats, _ = self.base(final_state)

        elif self.input_type == 'final_and_initial_state':
            i_state, f_state = trajectory
            i_feats, _ = self.base(i_state)
            f_feats, _ = self.base(f_state)
            obs_feats = torch.cat([i_feats, f_feats], 1)

        else:
            raise ValueError

        feats = self.fc(obs_feats)

        if self.option_space == 'discrete':
            opt_features = self.fc_logits(feats)
            if resizing_shape is not None:
                opt_features = opt_features.view(
                    *resizing_shape, *opt_features.shape[1:])
            dist = self.dist(opt_features)
            return dist

        else:
            concat_params = self.fc12(feats)

            mu, std = utils.get_mean_std(concat_params)
            if resizing_shape is not None:
                mu = mu.view(*resizing_shape, *mu.shape[1:])
                std = std.view(*resizing_shape, *std.shape[1:])

            gauss_dist = ds.normal.Normal(loc=mu, scale=std)

            return gauss_dist


class OptionsDecoder(nn.Module):
    """
    Module for decoding options given attributes and (optionally)
    initial state observation.

    Input: Attributes which can be either fully specified or
        partially specified. There are K attributes where the
        i-th attribute can take n_i values, i \in {0, 1, ... K-1}
        and (optionally) initial state observation.

    Output: Parameters of the probability distribution
        Q(\omega | A_k). Currently, a gaussian Q is supported i.e.
        location and scale of a gaussian are predicted as outputs.
    """
    def __init__(
        self,
        observability: str,
        input_type: str,
        encoder_type: str,
        obs_spaces: collections.OrderedDict,
        attr_embed_size: int,
        option_space: str,
        omega_option_dims: int,
        hidden_size: int,
        base_model: str,
        base_kwargs: Dict,
    ):
        '''
        Arguments:
            observability: Only 'full' is supported i.e. MDP setting

            input_type: 'goal_and_initial_state' or 'goal_only'; for
                taking as input the goal specification (attributes)
                and initial state in the first setting and just the
                goal specification in the second setting.

            encoder_type: 'single' or 'poe', the latter is a product
                of experts (POE) encoding which predicts the location
                and scale of each gaussian Q(\omega | A_k), where k is
                a specified attribute and computes the final probability
                as a multiplication of each predicted gaussian along
                with the prior P(\omega) which is N(0, I).

            input_attr_dim: Tuple specifying (n_0, n_1, ... n_K-1) i.e.
                the number of values for each attribute.

            attr_embed_size: See 'embed_size' in models.AttributeEmbedding

            hidden_size: Overloaded parameter specfying hidden size of
                MLPs, AttributeEmbedding objects and CNNs if any.

            agent_cfg_dims: The size of agent's config specification.
                This is needed when options are conditioned on the
                initial state observation which includes environment
                config and agent config.

            input_channels: The input channels of CNN used to encode
                environment config (cell map).
        '''
        super().__init__()

        assert input_type in ['goal_and_initial_state', 'goal_only']
        assert encoder_type in ['single', 'poe']
        assert 'mission' not in base_kwargs['obs_spaces']
        assert option_space in ['continuous', 'discrete']

        # self.agent_pos_dims = obs_spaces['agent_pos'].shape[0]
        self.input_attr_dims = obs_spaces['mission'].nvec
        self.attr_embed_size = attr_embed_size
        self.input_type = input_type
        self.encoder_type = encoder_type
        self.option_space = option_space
        self.omega_option_dims = omega_option_dims
        self.hidden_size = hidden_size
        self.observability = observability
        self.base_model = base_model

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        if base_kwargs is None:
            base_kwargs = {}

        if input_type == 'goal_and_initial_state':
            if observability == 'full':
                if self.base_model == 'mlp':
                    self.base = FlattenMLPBase(**base_kwargs)
                elif self.base_model == 'cnn-mlp':
                    self.base = CNNPlusMLPBase(**base_kwargs)
                else:
                    raise ValueError
            else:
                raise NotImplementedError

        # Only encode state observation if it is provided as input
        self.use_state_encoder = \
            self.input_type == 'goal_and_initial_state'

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2))

        # Encoding attributes
        if encoder_type == 'single':
            self.main_embed = AttributeEmbedding(
                embed_type='k-hot',
                input_attr_dims=self.input_attr_dims,
                embed_size=self.attr_embed_size,
                hidden_size=self.hidden_size,
                output_size=self.hidden_size)

            if self.use_state_encoder:
                self.fc = nn.Sequential(
                    init_(nn.Linear(hidden_size * 2, hidden_size)),
                    nn.LeakyReLU(0.1),
                )

            if self.option_space == 'continuous':
                self.fc12 = init_(nn.Linear(hidden_size,
                    2 * omega_option_dims))
            else:
                self.fc_logits = nn.Sequential(
                    init_(nn.Linear(hidden_size, hidden_size)),
                    nn.LeakyReLU(0.1),
                )
                self.dist = Categorical(hidden_size, self.omega_option_dims)

        elif encoder_type == 'poe':
            if self.option_space == 'discrete':
                raise NotImplementedError
            assert len(self.input_attr_dims) > 1, \
                "Use one-hot for single attribute"
            # assert output_size != None, \
            #     "Output size needed for k-hot embeddings"

            self.poe_embed = nn.ModuleList([
                nn.Sequential(
                    nn.Embedding(dim, hidden_size),
                    init_(nn.Linear(hidden_size, hidden_size)),
                    nn.LeakyReLU(0.1),
                    init_(nn.Linear(hidden_size, hidden_size)),
                    ) for dim in self.input_attr_dims])

            if self.use_state_encoder:
                self.fc_poe = nn.ModuleList([
                    nn.Sequential(
                        init_(nn.Linear(hidden_size + hidden_size,
                            hidden_size)),
                        nn.LeakyReLU(0.1),
                        init_(nn.Linear(hidden_size, hidden_size)),
                    ) for _ in self.input_attr_dims])

            self.fc12 = nn.ModuleList(
                    [init_(nn.Linear(hidden_size, 2 * omega_option_dims)) \
                        for _ in self.input_attr_dims])


    def forward(self, obs, do_sampling: bool,
                specifications=None):

        # target = inputs[:, :len(self.input_attr_dims)] #.long()

        # [NOTE] : Target will be -1 for unspecified attributes
        # [NOTE] : The unspecified attributes should be consistent
        # for all observations of the batch

        # specifications = (target != -1)
        # # specifications = np.where(target[0].cpu().numpy() != -1)[0]

        # if len(specifications) == 4:
        #     specifications = None

        # initial_obs = inputs[:, len(self.input_attr_dims):]

        if self.option_space == 'discrete':
            opt_features = self._encode(obs, specifications=specifications)
            dist = self.dist(opt_features)

            if do_sampling:
                option = dist.sample()
            else:
                option = dist.mode()

            option_log_probs = dist.log_probs(option)
            return option, dist, option_log_probs

        else:
            gauss_dist, hid = self._encode(obs, specifications=specifications)
            if do_sampling:
                dist_point = gauss_dist.rsample()
            else:
                dist_point = gauss_dist.mean
            dist = gauss_dist

            _ldj = 0.0
            return dist_point, gauss_dist, hid, _ldj


    def evaluate_options(self, obs, omega_option: torch.Tensor,
        specifications=None):
        if self.option_space == 'discrete':
            opt_features = self._encode(obs, specifications=specifications)
            dist = self.dist(opt_features)
            option_log_probs = dist.log_probs(omega_option)
            entropy = dist.entropy().mean()

        else:
            gauss_dist, hid = self._encode(obs, specifications=specifications)
            option_log_probs = gauss_dist.log_prob(omega_option)
            entropy = gauss_dist.entropy().mean()

        return option_log_probs, entropy

    def _encode(self, obs, specifications=None):
        if self.use_state_encoder:
            obs_feats, _ = self.base(obs)

        if self.encoder_type == 'single':
            # [NOTE] : We can evaluate recall for this case as well.
            # Options are:
            # 1. Predict a random value for missing attribute
            # 2. Do something else :P

            emb = self.main_embed(obs['mission'])

            if self.use_state_encoder:
                full_feats = torch.cat([emb, obs_feats], 1)
                hid = self.fc(full_feats)
            else:
                hid = emb

            if self.option_space == 'continuous':
                concat_params = self.fc12(hid)
                mu, std = utils.get_mean_std(concat_params)
                gauss_dist = ds.normal.Normal(loc=mu, scale=std)
            else:
                opt_features = self.fc_logits(hid)

        elif self.encoder_type == 'poe':
            '''
            Product of experts for composing gaussians of all
            specified attributes along with the prior.

            'specifications': mask tensor with ones for
                specified attributes and zeros otherwise
            '''

            # attr_indices = [0, 1, 2, 3]
            attr_indices = np.arange(len(self.input_attr_dims))
            if specifications is None:
                specifications = obs['mission'].new_ones(
                    (obs['mission'].shape[0], len(self.input_attr_dims)))
            # else:
            #     attr_indices = specifications
            #     assert len(attr_indices) >= 0

            # if target[:, attr_indices].min().item() < 0:
            #     import pdb; pdb.set_trace()
            #     pass

            mission = obs['mission'] * (torch.eq(specifications, 0).long())
            # obs['mission'].masked_fill_(torch.eq(specifications, 0), 0)

            # assert target[:, attr_indices].min().item() >= 0, \
            assert mission.min().item() >= 0, \
                "Negative index given as input to nn.embedding table"

            # Embed goals for specified attributes
            goal_embeds = [
                self.poe_embed[idx](mission[:, idx]) \
                    for idx in attr_indices]

            if self.use_state_encoder:
                # Forward pass goal embed and state observation
                cats = [self.fc_poe[attr_indices[idx]](
                    torch.cat([emb, obs_feats], 1)) \
                        for idx, emb in enumerate(goal_embeds)]
            else:
                cats = goal_embeds

            concat_params = [self.fc12[attr_indices[idx]](cat)\
                for idx, cat in enumerate(cats)]

            # Get mean and standard deviation
            concat_params = [utils.get_mean_std(par) \
                for par in concat_params]

            mus, stds = zip(*concat_params)

            # Initialize mu, std of prior before multiplying experts
            prior_mu = obs['agent_pos'].new_zeros(
                (obs['agent_pos'].shape[0], self.omega_option_dims))
            prior_std = obs['agent_pos'].new_ones(
                (obs['agent_pos'].shape[0], self.omega_option_dims))

            # Multiply gaussians to prior one by one in a for loop
            sum_sig = 1.0 / (prior_std ** 2)
            sum_mu = prior_mu / (prior_std ** 2)
            for idx, (mu, std) in enumerate(zip(mus, stds)):
                _mask = specifications[:, idx:idx+1].float()

                sum_sig += (_mask * 1.0) / (std ** 2)
                sum_mu += (_mask * mu) / (std ** 2)

            std_poe_sq = 1.0 / sum_sig
            std_poe = torch.sqrt(std_poe_sq)
            mu_poe = std_poe_sq * sum_mu

            # Get distributions object
            gauss_dist = ds.normal.Normal(loc=mu_poe, scale=std_poe)
            # return gauss_dist
            hid = None

        else:
            raise ValueError("Only 'single' and 'poe' supported")

        if self.option_space == 'continuous':
            return gauss_dist, hid
        else:
            return opt_features


class IBEncoder(nn.Module):
    def __init__(
        self,
        observability: str,
        latent_space: str,
        obs_spaces: collections.OrderedDict,
        # attr_embed_size,
        z_dims: int,
        z_std_clip_max: float,
        hidden_size: int,
        base_model: str,
        base_kwargs: Dict,
    ):
        super().__init__()

        # assert input_type in ['goal_and_initial_state']
        # assert encoder_type in ['single', 'poe']
        assert 'mission' not in base_kwargs['obs_spaces']
        # assert 'omega' in base_kwargs['obs_spaces']
        assert latent_space in ['gaussian']

        # self.encoder_type = encoder_type
        self.latent_space = latent_space
        self.z_dims = z_dims
        self.z_std_clip_max = z_std_clip_max
        self.hidden_size = hidden_size
        self.observability = observability
        self.base_model = base_model

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        if base_kwargs is None:
            base_kwargs = {}

        # if input_type == 'goal_and_initial_state':
        if observability == 'full':
            if self.base_model == 'mlp':
                self.base = FlattenMLPBase(**base_kwargs)
            elif self.base_model == 'cnn-mlp':
                self.base = CNNPlusMLPBase(**base_kwargs)
            else:
                raise ValueError
        else:
            raise NotImplementedError

        # # Only encode state observation if it is provided as input
        # self.use_state_encoder = \
        #     self.input_type == 'goal_and_initial_state'

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2))

        # Encoding attributes
        # if encoder_type == 'single':
        # self.fc = nn.Sequential(
        #     init_(nn.Linear(omega_option_dims, hidden_size)),
        #     nn.LeakyReLU(0.1),
        #     init_(nn.Linear(hidden_size, hidden_size)),
        # )

        # if self.use_state_encoder:
        # self.fc = nn.Sequential(
        #     init_(nn.Linear(hidden_size, hidden_size)),
        #     nn.LeakyReLU(0.1),
        # )

        if self.latent_space == 'gaussian':
            self.fc12 = init_(nn.Linear(hidden_size,
                2 * z_dims))

        elif self.latent_space == 'categorical':
            self.fc_logits = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)),
                nn.LeakyReLU(0.1),
            )
            self.dist = Categorical(hidden_size, self.z_dims)

        else:
            raise ValueError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, obs, rnn_hxs, masks, do_z_sampling: bool) \
        -> Tuple[torch.Tensor, torch.Tensor]:
        if self.latent_space == 'categorical':
            raise NotImplementedError
            opt_features, rnn_hxs = self._encode(obs, rnn_hxs=rnn_hxs)
            dist = self.dist(opt_features)

            if do_z_sampling:
                z_latent = dist.sample()
            else:
                z_latent = dist.mode()

            z_log_probs = dist.log_probs(z_latent)
            # return z_latent, dist, z_log_probs
            return z_latent, z_log_probs

        else:
            gauss_dist, hid, rnn_hxs = self._encode(
                obs, rnn_hxs=rnn_hxs, masks=masks)
            if do_z_sampling:
                z_latent = gauss_dist.rsample()
            else:
                z_latent = gauss_dist.mean

            z_log_prob = gauss_dist.log_prob(z_latent)
            # _ldj = 0.0
            return z_latent, z_log_prob, gauss_dist, rnn_hxs #, hid, _ldj


    def evaluate_options(self, **kwargs) -> Tuple[float, float]:
        '''Dummy function'''
        option_log_probs, option_entropy = 0.0, 0.0
        return option_log_probs, option_entropy


    # def evaluate_z_latents(self, obs, rnn_hxs, masks, z_latent: torch.Tensor):
    #     if self.latent_space == 'categorical':
    #         opt_features, rnn_hxs = self._encode(obs, rnn_hxs=rnn_hxs, masks=masks)
    #         dist = self.dist(opt_features)
    #         z_log_probs = dist.log_probs(z_latent)
    #         return z_latent, dist, z_log_probs, rnn_hxs
    #
    #     else:
    #         gauss_dist, hid, rnn_hxs = self._encode(
    #             obs, rnn_hxs=rnn_hxs, masks=masks)
    #         z_log_prob = gauss_dist.log_prob(z_latent)
    #         # _ldj = 0.0
    #         return z_latent, gauss_dist, z_log_prob, rnn_hxs

    def _encode(self, obs, rnn_hxs, masks):
        # if self.use_state_encoder:
        obs_feats, rnn_hxs = self.base(
            obs, rnn_hxs=rnn_hxs, masks=masks.clone())

        # if self.encoder_type == 'single':
        # full_feats = torch.cat([emb, obs_feats], 1)
        # hid = self.fc(full_feats)
        # hid = self.fc(obs_feats)
        hid = obs_feats

        if self.latent_space == 'gaussian':
            concat_params = self.fc12(hid)
            mu, std = utils.get_mean_std(concat_params)
            std = torch.clamp(std, max=self.z_std_clip_max)
            gauss_dist = ds.normal.Normal(loc=mu, scale=std)
            return gauss_dist, hid, rnn_hxs

        else:
            raise NotImplementedError
            opt_features = self.fc_logits(hid)
            return opt_features, rnn_hxs


class IBPolicy(Policy):
    def __init__(
        self,
        observability: str,
        action_dims: int,
        latent_space: str,
        obs_spaces: collections.OrderedDict,
        # attr_embed_size,
        z_dims: int,
        z_std_clip_max: float,
        hidden_size: int,
        base_model: str,
        base_kwargs: Dict,
        policy_base_kwargs: Dict,
    ):
        super().__init__(
            observability=observability,
            action_dims=action_dims,
            base_model=base_model,
            base_kwargs=policy_base_kwargs,
        )

        self.ib_encoder = IBEncoder(
            observability=observability,
            latent_space=latent_space,
            obs_spaces=obs_spaces,
            z_dims=z_dims,
            z_std_clip_max=z_std_clip_max,
            hidden_size=hidden_size,
            base_model=base_model,
            base_kwargs=base_kwargs,
        )

    @property
    def is_encoder_recurrent(self):
        return self.ib_encoder.base.is_recurrent

    @property
    def encoder_recurrent_hidden_state_size(self):
        """Size of ib_encoder rnn_hx."""
        return self.ib_encoder.base.recurrent_hidden_state_size

    def evaluate_z_latents(self, **kwargs):
        return self.ib_encoder.evaluate_z_latents(**kwargs)

    def encoder_forward(self, **kwargs):
        return self.ib_encoder.forward(**kwargs)

    def evaluate_options(self, **kwargs):
        return self.ib_encoder.evaluate_options(**kwargs)

    def get_action_dist(self, inputs, rnn_hxs, masks):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        dist_entropy = dist.entropy().mean()
        return value, dist, dist_entropy, rnn_hxs


class IBSupervisedEncoder(nn.Module):
    def __init__(
        self,
        observability: str,
        latent_space: str,
        obs_spaces: collections.OrderedDict,
        # attr_embed_size,
        z_dims: int,
        z_std_clip_max: float,
        goal_vector_obs_space,
        hidden_size: int,
        base_model: str,
        # base_kwargs: Dict,
    ):
        super().__init__()

        assert latent_space in ['gaussian']

        # self.encoder_type = encoder_type
        self.latent_space = latent_space
        self.z_dims = z_dims
        self.z_std_clip_max = z_std_clip_max
        self.hidden_size = hidden_size
        self.observability = observability
        assert len(goal_vector_obs_space.shape) == 1
        self.goal_vector_dims = goal_vector_obs_space.shape[0]
        # self.base_model = base_model

        self.base = base_model
        assert self.is_recurrent

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2))

        self.fc12 = init_(nn.Linear(hidden_size + self.goal_vector_dims,
            2 * z_dims))

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, obs, rnn_hxs, masks, do_z_sampling: bool) \
        -> Tuple[torch.Tensor, torch.Tensor]:

        gauss_dist, hid, rnn_hxs = self._encode(
            obs, rnn_hxs=rnn_hxs, masks=masks)
        if do_z_sampling:
            z_latent = gauss_dist.rsample()
        else:
            z_latent = gauss_dist.mean

        z_log_prob = gauss_dist.log_prob(z_latent)
        return z_latent, z_log_prob, gauss_dist, rnn_hxs


    def evaluate_options(self, **kwargs) -> Tuple[float, float]:
        '''Dummy function'''
        option_log_probs, option_entropy = 0.0, 0.0
        return option_log_probs, option_entropy

    def _encode(self, obs, rnn_hxs, masks):
        # if self.use_state_encoder:
        obs_feats, rnn_hxs = self.base(
            obs, rnn_hxs=rnn_hxs, masks=masks.clone())

        # if self.encoder_type == 'single':
        # full_feats = torch.cat([emb, obs_feats], 1)
        # hid = self.fc(full_feats)
        # hid = self.fc(obs_feats)
        hid = obs_feats

        goal_vector = obs['goal_vector']
        hid_cat = torch.cat([hid, goal_vector], 1)
        concat_params = self.fc12(hid_cat)
        mu, std = utils.get_mean_std(concat_params)
        std = torch.clamp(std, max=self.z_std_clip_max)
        gauss_dist = ds.normal.Normal(loc=mu, scale=std)
        return gauss_dist, hid, rnn_hxs


class IBSupervisedPolicy(Policy):
    def __init__(
        self,
        observability: str,
        action_dims: int,
        latent_space: str,
        obs_spaces: collections.OrderedDict,
        # attr_embed_size,
        z_dims: int,
        z_std_clip_max: float,
        hidden_size: int,
        base_model: str,
        base_kwargs: Dict,
        policy_base_kwargs: Dict,
    ):
        # assert 'goal_vector' not in policy_base_kwargs['obs_spaces']
        new_base_kwargs = policy_base_kwargs.copy()
        self.goal_vector_obs_space = policy_base_kwargs['obs_spaces']['goal_vector']
        new_base_kwargs['obs_spaces'].pop('goal_vector')

        super().__init__(
            observability=observability,
            action_dims=action_dims,
            base_model=base_model,
            base_kwargs=new_base_kwargs,
        )

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        # self.actor_net = init_(nn.Linear(hidden_size + z_dims,
            # 2 * z_dims))

        self.actor_net = nn.Sequential(
            init_(nn.Linear(hidden_size + z_dims, hidden_size)),
            nn.Tanh(),
        )

        self.critic_net = nn.Sequential(
            init_(nn.Linear(hidden_size + z_dims, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, 1)),
        )

        self.ib_encoder = IBSupervisedEncoder(
            observability=observability,
            latent_space=latent_space,
            obs_spaces=obs_spaces,
            z_dims=z_dims,
            goal_vector_obs_space=self.goal_vector_obs_space,
            z_std_clip_max=z_std_clip_max,
            hidden_size=hidden_size,
            base_model=self.base,
            # base_kwargs=base_kwargs,
        )
        assert 'goal_vector' not in self.base.obs_keys

    @property
    def is_encoder_recurrent(self):
        return self.ib_encoder.base.is_recurrent

    @property
    def encoder_recurrent_hidden_state_size(self):
        """Size of ib_encoder rnn_hx."""
        return self.ib_encoder.base.recurrent_hidden_state_size

    def evaluate_z_latents(self, **kwargs):
        return self.ib_encoder.evaluate_z_latents(**kwargs)

    def encoder_forward(self, **kwargs):
        return self.ib_encoder.forward(**kwargs)

    def evaluate_options(self, **kwargs):
        return self.ib_encoder.evaluate_options(**kwargs)

    def _state_z_forward(self, input_features):
        actor_features = self.actor_net(input_features)
        # value = self.critic_net(input_features.detach())
        value = self.critic_net(input_features)
        return value, actor_features

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        # value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        state_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        z_latent = inputs['z_latent']
        input_features = torch.cat([state_features, z_latent], 1)

        value, actor_features = self._state_z_forward(input_features)

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
        # value, _, _ = self.base(inputs, rnn_hxs, masks)
        state_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        z_latent = inputs['z_latent']
        input_features = torch.cat([state_features, z_latent], 1)
        value, actor_features = self._state_z_forward(input_features)

        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action,
        get_entropy=True):
        # value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        state_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        z_latent = inputs['z_latent']
        input_features = torch.cat([state_features, z_latent], 1)
        value, actor_features = self._state_z_forward(input_features)

        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        if get_entropy:
            dist_entropy = dist.entropy().mean()
        else:
            dist_entropy = None

        return value, action_log_probs, dist_entropy, rnn_hxs, dist

    def get_action_dist(self, inputs, rnn_hxs, masks):
        # value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        state_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        z_latent = inputs['z_latent']
        input_features = torch.cat([state_features, z_latent], 1)
        value, actor_features = self._state_z_forward(input_features)

        dist = self.dist(actor_features)
        dist_entropy = dist.entropy().mean()
        return value, dist, dist_entropy, rnn_hxs



# class IBSupervisedTDEncoder(nn.Module):
#     def __init__(
#         self,
#         observability: str,
#         latent_space: str,
#         obs_spaces: collections.OrderedDict,
#         # attr_embed_size,
#         z_dims: int,
#         z_std_clip_max: float,
#         goal_vector_obs_space,
#         hidden_size: int,
#         base_model: str,
#         # base_kwargs: Dict,
#     ):
#         super().__init__()
#
#         assert latent_space in ['gaussian']
#
#         # self.encoder_type = encoder_type
#         self.latent_space = latent_space
#         self.z_dims = z_dims
#         self.z_std_clip_max = z_std_clip_max
#         self.hidden_size = hidden_size
#         self.observability = observability
#         assert len(goal_vector_obs_space.shape) == 1
#         self.goal_vector_dims = goal_vector_obs_space.shape[0]
#         # self.base_model = base_model
#
#         self.base = base_model
#         assert self.is_recurrent
#
#         init_ = lambda m: init(m,
#             nn.init.orthogonal_,
#             lambda x: nn.init.constant_(x, 0),
#             np.sqrt(2))
#
#         self.fc12 = init_(nn.Linear(hidden_size + self.goal_vector_dims,
#             2 * z_dims))
#
#     @property
#     def is_recurrent(self):
#         return self.base.is_recurrent
#
#     @property
#     def recurrent_hidden_state_size(self):
#         """Size of rnn_hx."""
#         return self.base.recurrent_hidden_state_size
#
#     def forward(self, obs, rnn_hxs, masks, do_z_sampling: bool) \
#         -> Tuple[torch.Tensor, torch.Tensor]:
#
#         gauss_dist, hid, rnn_hxs = self._encode(
#             obs, rnn_hxs=rnn_hxs, masks=masks)
#         if do_z_sampling:
#             z_latent = gauss_dist.rsample()
#         else:
#             z_latent = gauss_dist.mean
#
#         z_log_prob = gauss_dist.log_prob(z_latent)
#         return z_latent, z_log_prob, gauss_dist, rnn_hxs
#
#
#     def evaluate_options(self, **kwargs) -> Tuple[float, float]:
#         '''Dummy function'''
#         option_log_probs, option_entropy = 0.0, 0.0
#         return option_log_probs, option_entropy
#
#     def _encode(self, obs, rnn_hxs, masks):
#         # if self.use_state_encoder:
#         obs_feats, rnn_hxs = self.base(
#             obs, rnn_hxs=rnn_hxs, masks=masks.clone())
#
#         # if self.encoder_type == 'single':
#         # full_feats = torch.cat([emb, obs_feats], 1)
#         # hid = self.fc(full_feats)
#         # hid = self.fc(obs_feats)
#         hid = obs_feats
#
#         goal_vector = obs['goal_vector']
#         hid_cat = torch.cat([hid, goal_vector], 1)
#         concat_params = self.fc12(hid_cat)
#         mu, std = utils.get_mean_std(concat_params)
#         std = torch.clamp(std, max=self.z_std_clip_max)
#         gauss_dist = ds.normal.Normal(loc=mu, scale=std)
#         return gauss_dist, hid, rnn_hxs


class IBSupervisedTDPolicy(Policy):
    def __init__(
        self,
        observability: str,
        action_dims: int,
        latent_space: str,
        obs_spaces: collections.OrderedDict,
        # attr_embed_size,
        z_dims: int,
        z_std_clip_max: float,
        hidden_size: int,
        base_model: str,
        base_kwargs: Dict,
        policy_base_kwargs: Dict,
    ):
        # assert 'goal_vector' not in policy_base_kwargs['obs_spaces']
        new_base_kwargs = policy_base_kwargs.copy()
        self._z_dims = z_dims
        self.goal_vector_obs_space = \
            policy_base_kwargs['obs_spaces']['goal_vector']
        self.goal_vector_dims = self.goal_vector_obs_space.shape[0]
        new_base_kwargs['obs_spaces'].pop('goal_vector')
        self.z_std_clip_max = z_std_clip_max

        super().__init__(
            observability=observability,
            action_dims=action_dims,
            base_model=base_model,
            base_kwargs=new_base_kwargs,
        )

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor_net = nn.Sequential(
            init_(nn.Linear(hidden_size + z_dims, hidden_size)),
            nn.Tanh(),
        )

        self.critic_net = nn.Sequential(
            init_(nn.Linear(hidden_size + z_dims, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, 1)),
        )

        self.z_enc_net = init_(nn.Linear(hidden_size \
            + self.goal_vector_dims, 2 * z_dims))

        # self.ib_encoder = IBSupervisedEncoder(
        #     observability=observability,
        #     latent_space=latent_space,
        #     obs_spaces=obs_spaces,
        #     z_dims=z_dims,
        #     goal_vector_obs_space=self.goal_vector_obs_space,
        #     z_std_clip_max=z_std_clip_max,
        #     hidden_size=hidden_size,
        #     base_model=self.base,
        #     # base_kwargs=base_kwargs,
        # )
        assert 'goal_vector' not in self.base.obs_keys

    @property
    def is_encoder_recurrent(self):
        return self.ib_encoder.base.is_recurrent

    @property
    def encoder_recurrent_hidden_state_size(self):
        """Size of ib_encoder rnn_hx."""
        return self.ib_encoder.base.recurrent_hidden_state_size

    # def evaluate_z_latents(self, **kwargs):
    #     return self.ib_encoder.evaluate_z_latents(**kwargs)
    #
    # def encoder_forward(self, **kwargs):
    #     return self.ib_encoder.forward(**kwargs)
    #
    # def evaluate_options(self, **kwargs):
    #     return self.ib_encoder.evaluate_options(**kwargs)

    @property
    def z_latent_size(self):
        return self._z_dims

    # def _state_z_forward(self, input_features):
    #     actor_features = self.actor_net(input_features)
    #     value = self.critic_net(input_features)
    #     return value, actor_features

    def act(
        self,
        inputs,
        rnn_hxs,
        masks,
        deterministic=False,
        do_z_sampling=True,
    ):
        # value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        state_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        # Z-Encoding
        z_latent, z_gauss_dist = self._z_encode(
            state_features, inputs['goal_vector'],
            do_z_sampling=do_z_sampling)

        actor_inp = torch.cat([state_features, z_latent], 1)
        actor_features = self.actor_net(actor_inp)
        value = self.critic_net(actor_inp)

        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return z_latent, z_gauss_dist, value, action, \
            action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        state_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        # Z-Encoding
        z_latent, z_gauss_dist = self._z_encode(
            state_features, inputs['goal_vector'], do_z_sampling=False)
        actor_inp = torch.cat([state_features, z_latent], 1)
        value = self.critic_net(actor_inp)
        return value

    def _z_encode(self, state_features, goal_vector, do_z_sampling):
        enc_input = torch.cat([state_features, goal_vector], 1)
        concat_params = self.z_enc_net(enc_input)
        mu, std = utils.get_mean_std(concat_params)
        std = torch.clamp(std, max=self.z_std_clip_max)
        z_gauss_dist = ds.normal.Normal(loc=mu, scale=std)
        if do_z_sampling:
            z_sample = z_gauss_dist.rsample()
        else:
            z_sample = z_gauss_dist.mean
        return z_sample, z_gauss_dist

    def evaluate_actions(
        self,
        inputs,
        z_eps,
        rnn_hxs,
        masks,
        action,
        get_entropy=True,
    ):
        # value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        state_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        enc_input = torch.cat([state_features, inputs['goal_vector']], 1)
        concat_params = self.z_enc_net(enc_input)
        mu, std = utils.get_mean_std(concat_params)
        std = torch.clamp(std, max=self.z_std_clip_max)
        z_gauss_dist = ds.normal.Normal(loc=mu, scale=std)
        z_sample = mu + (z_eps * std)

        actor_inp = torch.cat([state_features, z_sample], 1)

        actor_features = self.actor_net(actor_inp)
        value = self.critic_net(actor_inp)

        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        if get_entropy:
            dist_entropy = dist.entropy().mean()
        else:
            dist_entropy = None

        return z_sample, z_gauss_dist, value, action_log_probs, \
            dist_entropy, rnn_hxs, dist

    # def get_action_dist(self, inputs, rnn_hxs, masks):
    #     # value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
    #     state_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
    #     z_latent = inputs['z_latent']
    #     input_features = torch.cat([state_features, z_latent], 1)
    #     value, actor_features = self._state_z_forward(input_features)
    #
    #     dist = self.dist(actor_features)
    #     dist_entropy = dist.entropy().mean()
    #     return value, dist, dist_entropy, rnn_hxs

    def encoder_forward(self, obs, rnn_hxs, masks, do_z_sampling: bool):
        state_features, rnn_hxs = self.base(obs, rnn_hxs, masks)

        z_latent, gauss_dist = self._z_encode(
            state_features, obs['goal_vector'], do_z_sampling)

        z_log_prob = gauss_dist.log_prob(z_latent)
        # _ldj = 0.0
        return z_latent, z_log_prob, gauss_dist, rnn_hxs #, hid, _ldj

    def _encode(self, obs, rnn_hxs, masks):
        z_latent, z_log_prob, gauss_dist, rnn_hxs = \
            self.encoder_forward(obs, rnn_hxs, masks, do_z_sampling=False)
        return gauss_dist, None, rnn_hxs


class OptionsPolicy(Policy):
    def __init__(
        self,
        observability: str,
        option_dims: int,
        option_space: str,
        # obs_spaces: collections.OrderedDict,
        hidden_size: int,
        base_model: str,
        base_kwargs: Dict,
        # policy_base_kwargs: Dict,
    ):
        super().__init__(
            observability=observability,
            action_dims=option_dims,
            base_model=base_model,
            base_kwargs=base_kwargs,
        )
        assert option_space in ['continuous', 'discrete']
        self.option_space = option_space

        if self.option_space == 'continuous':
            # Overwrite Policy class attributes
            # self.fc12 = init_(nn.Linear(hidden_size,
            #     2 * omega_option_dims))
            self.dist = DiagGaussian(self.base.output_size, option_dims)

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if self.option_space == 'discrete':
            return super().act(inputs, rnn_hxs, masks, deterministic)
        else:
            value, actor_features, rnn_hxs = self.base(
                inputs, rnn_hxs, masks.clone())
            # assert torch.isnan(actor_features).sum().item() == 0
            dist = self.dist(actor_features)

            if deterministic:
                action = dist.mean
            else:
                action = dist.rsample()

            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy().mean()

            return value, action, action_log_probs, rnn_hxs


class DummyModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dummy_net = nn.Linear(2, 2)

    def evaluate_options(self, **kwargs):
        return 0.0, 0.0
