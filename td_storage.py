import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorageTD(object):
    __slots__ = [
        'obs',
        'prev_final_obs',
        'prev_final_mask',
        'prev_final_visit_count',
        'prev_final_heur_ds',
        # 'option_reset_masks',
        'recurrent_hidden_states',
        # 'traj_rnn_hxs',
        'ib_enc_hidden_states',
        'rewards',
        'value_preds',
        'next_value',
        'returns',
        'action_log_probs',
        'actions',
        'masks',
        # 'omega_option_dims',
        # 'omega_option_steps',
        # 'option_log_probs',
        # 'options_rhx',
        # 'option_values',
        # 'option_intervals',
        # 'next_option_value',
        'num_steps',
        'step',
        'omega_option',
        'omega_eps',
        'z_latents',
        'z_eps',
        'z_dists',
        'z_logprobs',
        'use_gae',
        'gamma',
        'tau',
    ]
    def __init__(self,
                num_steps,
                num_processes,
                obs_spaces,
                use_gae,
                gamma,
                tau,
                recurrent_hidden_state_size,
                # traj_rhx_size,
                # omega_option_dims=None,
                z_latent_dims=10):

        self.obs = {}
        self.prev_final_obs = {}
        for key, space in obs_spaces.items():
            dtype = space.dtype
            if dtype == np.int8:
                dtype = np.int64
            elif dtype == np.float64:
                dtype = np.float32

            self.obs[key] = torch.zeros(num_steps + 1, num_processes,
                *space.shape).type_as(torch.from_numpy(
                np.zeros(1, dtype=dtype)))
            self.prev_final_obs[key] = torch.zeros(num_steps, num_processes,
                *space.shape).type_as(torch.from_numpy(
                np.zeros(1, dtype=dtype)))

        self.obs = DictObs(self.obs)
        self.prev_final_obs = DictObs(self.prev_final_obs)
        self.prev_final_mask = torch.zeros(num_steps, num_processes, 1)
        self.prev_final_visit_count = torch.zeros(num_steps, num_processes, 1)
        self.prev_final_heur_ds = torch.zeros(num_steps, num_processes, 1)
        # self.option_reset_masks = torch.zeros(num_steps, num_processes, 1)
        # self.obs = DictObs({key:torch.zeros(
        #     num_steps + 1, num_processes, *space.shape) \
        #         for key, space in obs_spaces.items()})

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        # self.traj_rnn_hxs = torch.zeros(
        #     num_steps + 1, num_processes, traj_rhx_size)
        self.ib_enc_hidden_states = []
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.next_value = torch.zeros(num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        # self.action_log_prob_sum = torch.zeros(num_steps, num_processes, 1)
        # if action_space.__class__.__name__ == 'Discrete':
        action_shape = 1
        # else:
        #     action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        # if action_space.__class__.__name__ == 'Discrete':
        self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # self.omega_option_dims = omega_option_dims
        # self.omega_option = torch.zeros(num_steps, num_processes, omega_option_dims)
        # self.omega_eps = torch.zeros(num_steps, num_processes, omega_option_dims)
        # # self.omega_option = None
        # self.option_intervals = []
        # self.omega_option_steps = []
        # self.option_log_probs = []
        # self.option_values = []
        # self.options_rhx = []
        # self.next_option_value = None
        # self.z_latents = torch.zeros(num_steps, num_processes, z_latent_dims)
        self.z_eps = torch.zeros(num_steps, num_processes, z_latent_dims)
        # self.z_latents = []
        # self.z_dists = []
        # self.z_logprobs = torch.zeros(num_steps, num_processes, z_latent_dims)

        self.num_steps = num_steps
        self.step = 0

        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = tau

    def to(self, device):
        self.obs = self.obs.to(device)
        self.prev_final_obs = self.prev_final_obs.to(device)
        self.prev_final_mask = self.prev_final_mask.to(device)
        self.prev_final_visit_count = self.prev_final_visit_count.to(device)
        self.prev_final_heur_ds = self.prev_final_heur_ds.to(device)
        # self.option_reset_masks = self.option_reset_masks.to(device)
        # self.omega_option = self.omega_option.to(device)
        # self.omega_eps = self.omega_eps.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.z_eps = self.z_eps.to(device)
        # self.traj_rnn_hxs = self.traj_rnn_hxs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self,
               obs,
               # omega_option,
               # omega_eps,
               recurrent_hidden_states,
               # traj_rnn_hxs,
               actions,
               action_log_probs,
               value_preds,
               rewards,
               masks,
               z_eps=None,
               # option_reset_masks,
        ):
        self.obs[self.step + 1].copy_(obs)
        # self.omega_option[self.step].copy_(omega_option)
        # self.omega_eps[self.step].copy_(omega_eps)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        # self.traj_rnn_hxs[self.step + 1].copy_(traj_rnn_hxs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        # self.option_reset_masks[self.step].copy_(option_reset_masks)

        if z_eps is not None:
            self.z_eps[self.step].copy_(z_eps)

        self.step = (self.step + 1) % self.num_steps

    def insert_next_value(self, next_value):
        self.next_value.copy_(next_value)

    def insert_next_option_value(self, next_option_value):
        self.next_option_value = next_option_value

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        # self.traj_rnn_hxs[0].copy_(self.traj_rnn_hxs[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(
        self,
        rewards,
        masks,
        value_preds,
        next_value,
        step_additives,
        # episodic_additives,
    ):
        num_steps, num_processes, _ = rewards.shape
        returns = rewards.new_zeros(num_steps + 1, num_processes, 1)

        if self.use_gae:
            value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(rewards.size(0))):
                delta = rewards[step] + step_additives[step] +\
                    self.gamma * value_preds[step + 1] * masks[step + 1] - \
                    value_preds[step]

                gae = delta\
                    + self.gamma * self.tau * masks[step + 1] * gae

                returns[step] = gae + value_preds[step]
        else:
            returns[-1] = next_value
            for step in reversed(range(rewards.size(0))):
                returns[step] = returns[step + 1] * \
                    self.gamma * masks[step + 1] + \
                    rewards[step] + step_additives[step]

        # returns[:-1] += (episodic_additives * masks[:-1])
        # returns += (episodic_additives * masks)
        return returns

    # def compute_returns(self,
    #                     next_value,
    #                     use_gae,
    #                     gamma,
    #                     tau,
    #                     use_proper_time_limits=True):
    #     if use_gae:
    #         self.value_preds[-1] = next_value
    #         gae = 0
    #         for step in reversed(range(self.rewards.size(0))):
    #             delta = self.rewards[step] \
    #                 + gamma * self.value_preds[step + 1] * self.masks[step + 1] \
    #                 - self.value_preds[step]
    #
    #             gae = delta \
    #                 + gamma * tau * self.masks[step + 1] * gae
    #
    #             self.returns[step] = gae + self.value_preds[step]
    #     else:
    #         self.returns[-1] = next_value
    #         for step in reversed(range(self.rewards.size(0))):
    #             self.returns[step] = \
    #                 self.returns[step + 1] * gamma * self.masks[step + 1] \
    #                 + self.rewards[step]


def obs_stack_helper(obs, dim):
    return DictObs({key:torch.stack([item[key] for item in obs], dim) \
        for key in obs[0].keys()})

class DictObs(object):
    __slots__ = ['obs']
    def __init__(self, obs):
        self.obs = obs

    def __str__(self):
        return self.obs.__str__()

    def __repr__(self):
        return self.obs.__repr__()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.obs[index]
        else:
            return DictObs({key:self.obs[key][index] for key in self.obs.keys()})

    def __setitem__(self, key, value):
            self.obs[key] = value

    def __add__(self, other):
        if isinstance(other, DictObs) or isinstance(other, dict):
            return DictObs(
                {key: self.obs[key] + other[key] for key in self.obs.keys()})
        else:
            return DictObs(
                {key: self.obs[key] + other for key in self.obs.keys()})

    def __mul__(self, other):
        if isinstance(other, DictObs) or isinstance(other, dict):
            return DictObs(
                {key: expanded_mul(self.obs[key], other[key]) \
                    for key in self.obs.keys()})
        else:
            return DictObs(
                {key: expanded_mul(self.obs[key], other) \
                    for key in self.obs.keys()})

    def __rmul__(self, other):
        return self.__mul__(other)

    def mul(self, other):
        return self.__mul__(other)

    def view(self, *args):
        return DictObs({key:self.obs[key].view(*args) for key in self.obs.keys()})

    def to(self, device):
        for key in self.obs:
            self.obs[key] = self.obs[key].to(device)
        return self

    @property
    def shape(self):
        return {key:self.obs[key].shape for key in self.obs}

    def keys(self):
        return self.obs.keys()

    def values(self):
        return self.obs.values()

    def items(self):
        return self.obs.items()

    def pop(self, key, default=None):
        self.obs.pop(key, default)

    def update(self, dict):
        for key, value in dict.items():
            self.obs[key] = value

    def copy_(self, obs):
        for key in self.obs:
            self.obs[key].copy_(obs[key])

    def copy(self):
        return DictObs(self.obs.copy())

    def unsqueeze(self, dim):
        return DictObs({key:self.obs[key].unsqueeze(dim) for key in self.obs})

    def repeat_dim(self, dim, reps):
        obs = {}
        for key in self.obs:
            shape = self.obs[key].shape
            ones_shape = [1] * len(shape)
            ones_shape[dim] = reps
            obs[key] = self.obs[key].repeat(*ones_shape)
        return DictObs(obs)

    def clone(self):
        cloned_obs = {}
        for key in self.obs:
            cloned_obs[key] = self.obs[key].clone()
        return DictObs(cloned_obs)

    def flatten_two(self):
        return DictObs({key:self.obs[key].view(-1,
            *self.obs[key].size()[2:]) for key in self.obs})

    def flatten_three(self):
        return DictObs({key:self.obs[key].view(-1,
            *self.obs[key].size()[3:]) for key in self.obs})

    def unflatten_two(self, dim0, dim1):
        return DictObs({key:self.obs[key].view(dim0, dim1, \
            *self.obs[key].size()[1:]) for key in self.obs})

    def fill_(self, value):
        for key in self.obs:
            self.obs[key].fill_(value)

    def gather_timesteps(self, indices):
        '''
        Input 'indices' are assumed to be of shape (T, B),
        where T is the time steps dim (= 0), B is num_processes;
        indices should lie in range(0, max-time-steps)
        '''
        output = {}
        for key in self.obs:
            extra_dims = len(self.obs[key].shape) - (len(indices.shape) + 1)
            _tail = [1] * extra_dims
            _indices = indices.view(*indices.shape, *_tail)
            indices_expanded = _indices.expand_as(self.obs[key])
            output[key] = self.obs[key].gather(
                dim=0, index=indices_expanded[:1])[0]

        return DictObs(output)

def expanded_mul(tensor1, tensor2):
    def expand_other(target, other):
        for _ in range(target.dim() - other.dim()):
            other = other.unsqueeze(-1)
        return other

    if tensor1.dim() == tensor2.dim():
        return tensor1 * tensor2
    elif tensor1.dim() > tensor2.dim():
        tensor2 = expand_other(tensor1, tensor2)
    else:
        tensor1 = expand_other(tensor2, tensor1)
    return tensor1 * tensor2
