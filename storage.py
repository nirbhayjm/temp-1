import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    __slots__ = [
        'obs',
        'recurrent_hidden_states',
        'ib_enc_hidden_states',
        'rewards',
        'value_preds',
        'next_value',
        'returns',
        'action_log_probs',
        'actions',
        'masks',
        'omega_option_dims',
        'omega_option_steps',
        'option_log_probs',
        'options_rhx',
        'option_values',
        'option_intervals',
        'next_option_value',
        'num_steps',
        'step',
        'omega_option',
        'z_latents',
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
                omega_option_dims=None,
                z_latent_dims=10):

        self.obs = {}
        for key, space in obs_spaces.items():
            dtype = space.dtype
            if dtype == np.int8:
                dtype = np.int64
            elif dtype == np.float64:
                dtype = np.float32
            self.obs[key] = torch.zeros(num_steps + 1, num_processes,
                *space.shape).type_as(torch.from_numpy(
                np.zeros(1, dtype=dtype)))
        self.obs = DictObs(self.obs)
        # self.obs = DictObs({key:torch.zeros(
        #     num_steps + 1, num_processes, *space.shape) \
        #         for key, space in obs_spaces.items()})

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
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

        self.omega_option_dims = omega_option_dims
        # self.omega_option = torch.zeros(num_steps, num_processes, omega_option_dims)
        self.omega_option = None
        self.option_intervals = []
        self.omega_option_steps = []
        self.option_log_probs = []
        self.option_values = []
        self.options_rhx = []
        self.next_option_value = None
        # self.z_latents = torch.zeros(num_steps, num_processes, z_latent_dims)
        self.z_latents = []
        self.z_dists = []
        # self.z_logprobs = torch.zeros(num_steps, num_processes, z_latent_dims)

        self.num_steps = num_steps
        self.step = 0

        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = tau

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self,
               obs,
               recurrent_hidden_states,
               actions,
               action_log_probs,
               value_preds,
               rewards,
               masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    # def insert_option(self, omega_option):
    #     self.omega_option.copy_(omega_option)

    def insert_option(self, omega):
        self.omega_option = omega.clone()

    def insert_option_t(self, step, omega_option_t, option_log_probs,
        option_value, options_rhx):
        self.option_intervals.append(step)
        self.omega_option_steps.append(omega_option_t)
        self.option_log_probs.append(option_log_probs)
        self.option_values.append(option_value)
        self.options_rhx.append(options_rhx)

    def insert_next_value(self, next_value):
        self.next_value.copy_(next_value)

    def insert_next_option_value(self, next_option_value):
        self.next_option_value = next_option_value

    def insert_z_latent(self, z_latent, z_logprobs, z_dist,
        ib_enc_hidden_states=None):
        # self.z_latents[self.step].copy_(z_latent)
        self.z_latents.append(z_latent)
        self.z_dists.append(z_dist)
        # self.z_logprobs[self.step].copy_(z_logprobs)
        if ib_enc_hidden_states is not None:
            self.ib_enc_hidden_states.append(ib_enc_hidden_states)

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])

    def reset_storage(self):
        self.obs.fill_(0)
        self.recurrent_hidden_states.fill_(0)
        self.rewards.fill_(0)
        self.value_preds.fill_(0)
        self.returns.fill_(0)
        self.next_value.fill_(0)
        self.action_log_probs.fill_(0)
        self.actions.fill_(0)
        self.masks.fill_(1)
        # self.z_latents.fill_(0)
        self.z_latents = []
        self.z_dists = []
        self.ib_enc_hidden_states = []
        self.omega_option_steps = []
        self.option_log_probs = []
        self.option_values = []
        self.options_rhx = []
        self.option_intervals = []
        self.next_option_value = None
        # self.z_logprobs.fill_(0)
        # self.omega_option.fill_(0)
        self.omega_option = None
        self.step = 0

    def compute_returns(
        self,
        rewards,
        masks,
        value_preds,
        next_value,
        step_additives,
        episodic_additives,
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
        returns += (episodic_additives * masks)
        return returns

    def feed_forward_generator(self, advantages, num_mini_batch):
        raise NotImplementedError("Need to remove z_latents")
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)

        omega_option_tiled = self.omega_option.\
            unsqueeze(0).repeat(num_steps,1,1)

        for indices in sampler:
            obs_batch = self.obs[:-1].flatten_two()[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1,
                self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            omega_batch = self.omega_option.view(
                -1, self.omega_option.size(-1))[indices]
            z_batch = self.z_latents.view(-1, self.z_latents.size(-1))[indices]
            z_logprobs = self.z_logprobs.view(-1, self.z_logprobs.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]
            omega_option_batch = omega_option_tiled.view(-1, self.omega_option_dims)[indices]


        yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
            return_batch, masks_batch, old_action_log_probs_batch,\
            adv_targ, omega_option_batch, omega_batch, z_batch, z_logprobs

    def recurrent_generator(self, advantages, num_mini_batch):
        raise NotImplementedError("Need to remove z_latents")
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        assert num_processes % num_mini_batch == 0, (
            "recurrent_generator requires num_processes to"
            "be divisible by num_mini_batch")
        num_envs_per_batch = num_processes // num_mini_batch

        omega_option_tiled = self.omega_option.\
            unsqueeze(0).repeat(self.num_steps,1,1)

        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            omega_batch = []
            z_batch = []
            z_logprobs = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            omega_option_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                omega_batch.append(self.omega_option[:, ind])
                z_batch.append(self.z_latents[:, ind])
                z_logprobs.append(self.z_logprobs[:, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

                omega_option_batch.append(omega_option_tiled[:, ind])
            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            # obs_batch = torch.stack(obs_batch, 1)
            obs_batch = obs_stack_helper(obs_batch, dim=1)
            actions_batch = torch.stack(actions_batch, 1)
            omega_batch = torch.stack(omega_batch, 1)
            z_batch = torch.stack(z_batch, 1)
            z_logprobs = torch.stack(z_logprobs, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            omega_option_batch = torch.stack(omega_option_batch, 1)
            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            # obs_batch = _flatten_helper(T, N, obs_batch)
            obs_batch = obs_batch.flatten_two()
            actions_batch = _flatten_helper(T, N, actions_batch)
            omega_batch = _flatten_helper(T, N, omega_batch)
            z_batch = _flatten_helper(T, N, z_batch)
            z_logprobs = _flatten_helper(T, N, z_logprobs)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            omega_option_batch = _flatten_helper(T, N, omega_option_batch)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ, omega_option_batch, omega_batch, z_batch, z_logprobs


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
                {key: self.obs[key] * other[key] for key in self.obs.keys()})
        else:
            return DictObs(
                {key: self.obs[key] * other for key in self.obs.keys()})

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
