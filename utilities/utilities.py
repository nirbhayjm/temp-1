import math
import numpy as np

import gym

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchnet.meter.movingaveragevaluemeter import MovingAverageValueMeter

import cv2

def confirm(prompt=None, resp=False):
    """prompts for yes or no response from the user. Returns True for yes and
    False for no.

    'resp' should be set to the default value assumed by the caller when
    user simply types ENTER.

    >>> confirm(prompt='Create Directory?', resp=True)
    Create Directory? [y]|n:
    True
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y:
    False
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y: y
    True

    """

    if prompt is None:
        prompt = 'Confirm'

    if resp:
        prompt = '%s [%s]|%s: ' % (prompt, 'y', 'n')
    else:
        prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')

    while True:
        ans = input(prompt)
        if not ans:
            return resp
        if ans not in ['y', 'Y', 'n', 'N']:
            print('please enter y or n.')
            continue
        if ans == 'y' or ans == 'Y':
            return True
        if ans == 'n' or ans == 'N':
            return False

# class RewardLogger(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.episode_reward = 0
#
#     def step(self, *args, **kwargs):
#         obs, reward, done, info = self.env.step(*args, **kwargs)
#
#         self.episode_reward += reward
#         if done:
#             info['episode_reward'] = self.episode_reward
#
#         return obs, reward, done, info
#
#     def reset(self, *args, **kwargs):
#         self.episode_reward = 0
#         return self.env.reset(*args, **kwargs)

def make_env(env_id, seed, rank, config_seed=None):
    def _thunk():
        env = gym.make(env_id)
        # env = RewardLogger(env)

        env.seed(seed + rank)
        if config_seed is not None:
            env.seed_config(config_seed + rank)
        return env

    return _thunk

class MovingAverage(MovingAverageValueMeter):
    __slots__ = [
        'windowsize',
        'input_shape',
        'valuequeue',
        'sum',
        'var',
        'n',
    ]
    def __init__(self, window_size):
        self.windowsize = window_size
        self.valuequeue = np.zeros(self.windowsize)
        self.reset()

    # def value(self):
    #     mean, std = super().value()
    #     _n = min(self.n, self.windowsize)
    #     sem = std / _n
    #     return mean, sem

    def reset(self):
        self.sum = 0.0
        self.n = 0
        self.var = 0.0
        self.valuequeue = np.zeros(self.windowsize)


class MovingAverageMoment(object):
    __slots__ = [
        'use_median',
        'mean_meter',
        'max_meter',
        'min_meter',
        'median_meter',
    ]
    def __init__(self, window_size, use_median=False):
        self.use_median = use_median
        self.mean_meter = MovingAverage(window_size)
        self.max_meter = MovingAverage(window_size)
        self.min_meter = MovingAverage(window_size)
        if self.use_median:
            self.median_meter = MovingAverage(window_size)

    def add(self, value_array):
        self.mean_meter.add(value_array.mean())
        self.max_meter.add(value_array.max())
        self.min_meter.add(value_array.min())
        if self.use_median:
            self.median_meter.add(np.median(value_array))

    def value(self):
        out = {
            'mean': self.mean_meter.value(),
            'max': self.max_meter.value(),
            'min': self.min_meter.value(),
        }
        if self.use_median:
            out['median'] =  self.median_meter.value()
        return out

class MovingAverageHeatMap(MovingAverage):
    __slots__ = [
        'windowsize',
        'input_shape',
        'valuequeue',
        'freq_normalize',
        'sum',
        'var',
        'n',
    ]
    def __init__(self, input_shape, windowsize, freq_normalize=False):
        self.windowsize = windowsize
        self.input_shape = input_shape
        self.freq_normalize = freq_normalize
        self.reset()

    def value(self):
        n = min(self.n, self.windowsize)
        if self.freq_normalize:
            mean = normalize_map(self.valuequeue)
        else:
            mean = self.sum / max(1, n)
        # std = math.sqrt(max((self.var - n * (mean ** 2)) / max(1, n - 1), 0))
        return mean #, std

    def add(self, value):
        queueid = (self.n % self.windowsize)
        oldvalue = self.valuequeue[queueid]
        self.sum += value - oldvalue
        # self.var += (value ** 2) - (oldvalue ** 2)
        self.valuequeue[queueid] = value
        self.n += 1

    def reset(self):
        self.sum = np.zeros(self.input_shape)
        self.n = 0
        # self.var = np.zeros(self.windowsize)
        self.valuequeue = np.zeros((self.windowsize, *self.input_shape))

class DictArrayQueue(object):
    __slots__ = [
        'windowsize',
        'queue',
        'n',
    ]
    def __init__(self, window_size, shape_dict):
        self.windowsize = window_size
        self.queue = {}
        for key, shape in shape_dict.items():
            self.queue[key] = -np.ones((window_size, *shape))
        self.reset()

    def reset(self):
        self.n = 0
        for key in self.queue.keys():
            self.queue[key] *=0
            self.queue[key] -= 1

    def add(self, value_dict):
        for key in value_dict.keys():
            value = value_dict[key]
            queueid = (self.n % self.windowsize)
            # oldvalue = self.queue[key][queueid]
            self.queue[key][queueid] = value
        self.n += 1

    def get_all_items(self):
        assert self.n > 0
        output_queue = {}
        if self.n < self.windowsize - 1:
            for key in self.queue.keys():
                output_queue[key] = self.queue[key][:self.n].copy()
        else:
            for key in self.queue.keys():
                output_queue[key] = self.queue[key].copy()
        return output_queue


def freeze_params(net):
    for param in net.parameters():
        param.requires_grad = False

def load_args(args, session_params, to_load_params, ckpt_path, verbose=True):
    print("Loading args from ckpt: {}".format(ckpt_path))
    load_ckpt = torch.load(ckpt_path)
    loaded_args = load_ckpt['params']
    if type(to_load_params) == str and to_load_params == 'all':
        if verbose: print("Loading all params!")
        LOAD_ALL = True
    else:
        LOAD_ALL = False

    for param in loaded_args:
        if param not in session_params:
            if hasattr(args, param):
                old_val = getattr(args, param)
                if old_val != loaded_args[param]:
                    if type(args) == dict:
                        args[param] = loaded_args[param]
                    else:
                        setattr(args, param, loaded_args[param])
                    if verbose:
                        print("Replacing arg from ckpt: {} - {} -> {}".format(
                            param, old_val, loaded_args[param]))
            else:
                if param in to_load_params or LOAD_ALL:
                    if type(args) == dict:
                        args[param] = loaded_args[param]
                    else:
                        setattr(args, param, loaded_args[param])
                    if verbose:
                        print("Setting arg from ckpt: {} -> {}".format(
                            param, loaded_args[param]))
                else:
                    if verbose:
                        print("Skipping param {}".format(param))
    print("Loaded args!")
    return args, loaded_args['iter_id']

def load_from_ckpt(
    # agent,
    device,
    actor_critic,
    # options_decoder,
    ckpt_path,
    trajectory_encoder=None,
    trajectory_optim=None,
):
    # print("Loading actor_critic model from ckpt: {}".format(
        # ckpt_path))
    load_ckpt = torch.load(ckpt_path)
    # actor_critic_mdict = {key:ac_mdict[key] for key in ac_mdict \
    #     if 'options_decoder' not in key}
    # options_decoder_mdict = {key:od_mdict[key] for key in od_mdict \
    #     if 'options_decoder' not in key}

    actor_critic.load_state_dict(load_ckpt['model'], strict=False)
    # options_decoder.load_state_dict(load_ckpt['options_decoder'], strict=False)

    actor_critic.to(device)
    # options_decoder.to(device)

    # if args.model == 'hier':
    if trajectory_encoder is not None:
        trajectory_encoder.load_state_dict(load_ckpt['trajectory_encoder'])
        trajectory_optim.load_state_dict(load_ckpt['trajectory_optim'])
    #     if args.hier_mode != 'transfer' and agent.options_policy is not None:
    #         agent.options_policy_optim.load_state_dict(load_ckpt['options_policy_optim'])
    #     agent.actor_critic_optim.load_state_dict(load_ckpt['actor_critic_optim'])
    #     trajectory_encoder.to(device)
    # else:
    #     agent.optimizer.load_state_dict(load_ckpt['optimizer'])

def q_start_curriculum(iter_id,
                       iters_per_epoch,
                       start_after_epochs):
    epoch_id = (1.0 * iter_id) / iters_per_epoch

    if epoch_id < start_after_epochs:
        return False
    else:
        return True

def kl_coefficient_curriculum(iter_id,
                              iters_per_epoch,
                              start_after_epochs,
                              linear_growth_epochs):
    '''
    Curriculum to anneal in KL loss coefficient.
    Linear annealing i.e. growth starts after and grows over
    number of epochs specified by 'start_after_epochs' and
    'linear_growth_epochs' respectively.

                _____ 1.0
               /
       0.0 ___/
    '''

    epoch_id = (1.0 * iter_id) / iters_per_epoch

    if epoch_id < start_after_epochs:
        return 0.0

    elif epoch_id < start_after_epochs + linear_growth_epochs:
        return (epoch_id - start_after_epochs) / linear_growth_epochs

    else:
        return 1.0

def omega_dims_curriculum(traj_enc_ll,
                          threshold,
                          current_omega,
                          max_omega,
                          growth_ratio):

    if current_omega < max_omega and traj_enc_ll > threshold:
        new_omega = min(int((growth_ratio * current_omega) + 1), max_omega)
        print("Omega growth {} -> {}".format(current_omega, new_omega))
        return new_omega
    else:
        return current_omega


def get_mean_std(concat_params):
    """Given concatenated mean, softplus-inverse of std, get mean and std.
    Args:
        concat_params: A torch `Tensor` of [N, H, .., D]
    Returns:
        mu: A torch `Tensor` of [N, H, .., D/2]
        std: A torch `Tensor` of [N, H, .., D/2], non-negative
    """
    if concat_params.dim() <= 1:
        raise ValueError('Input must be of rank greater than 1.')
    mu, inv_softplus_std = concat_params.split(
        int(concat_params.size(-1)/2), dim=-1)

    std = torch.nn.Softplus()(inv_softplus_std)
    return mu, std

def flatten_batch_dims(*args):
    flattened_args = []

    for tensor in args:
        eff_dim = tensor.shape[0] * tensor.shape[1]
        flattened_args.append(tensor.reshape((eff_dim,) + tensor.shape[2:]))

    return flattened_args

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    if type(dilation) is not tuple:
        dilation = (dilation, dilation)

    h = (h_w[0] + (2 * pad[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    return h, w

def conv_sequential_output_shape(h_w ,conv_seq):
    assert isinstance(conv_seq, torch.nn.modules.container.Sequential)

    # print("h_w initial: {}".format(h_w))
    for _module in conv_seq:
        if isinstance(_module, torch.nn.modules.conv.Conv2d)\
        or isinstance(_module, torch.nn.modules.pooling.MaxPool2d):
            h_w = conv_output_shape(
                h_w = h_w,
                kernel_size=_module.kernel_size,
                stride=_module.stride,
                pad=_module.padding,
                dilation=_module.dilation,
            )
            if hasattr(_module, 'out_channels'):
                _out_filters = _module.out_channels
            # print("h_w: {}".format(h_w))
        # elif isinstance(_module, torch.nn.modules.pooling.MaxPool2d):
        #     import pdb; pdb.set_trace()
        #     _module.

    return h_w, _out_filters

def weight_initialize(root, itype='xavier'):
    assert itype == 'xavier', 'Only Xavier initialization supported'

    for module in root.modules():
        # Initialize weights
        name = type(module).__name__
        # If linear or embedding
        if name in ['Embedding', 'Linear']:
            fanIn = module.weight.data.size(0)
            fanOut = module.weight.data.size(1)

            factor = math.sqrt(2.0 / (fanIn + fanOut))
            weight = torch.randn(fanIn, fanOut) * factor
            module.weight.data.copy_(weight)
        elif 'LSTM' in name:
            for name, param in module.named_parameters():
                if 'bias' in name:
                    param.data.fill_(0.0)
                else:
                    fanIn = param.size(0)
                    fanOut = param.size(1)

                    factor = math.sqrt(2.0 / (fanIn + fanOut))
                    weight = torch.randn(fanIn, fanOut) * factor
                    param.data.copy_(weight)
        else:
            pass

        # Check for bias and reset
        if hasattr(module, 'bias') and type(module.bias) != bool:
            module.bias.data.fill_(0.0)

def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None

def normalize_map(map):
    counts = (map > 0).astype('float').sum(0)
    return map.sum(0, keepdims=False) / (counts + 1e-10)

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


def update_current_obs(obs, current_obs, obs_shape, num_stack):
    shape_dim0 = obs_shape[0]
    if num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs


def draw_map(agent_cfg, target_xy, cell_map, done_flag=None, canvas_size=100):
    # cell_map = cell_map_flat.view(cell_map_flat.size(0), 30, 30)
    # cell_map = cell_map.cpu().numpy()
    agent_cfg = agent_cfg #.cpu().numpy()
    target_xy = target_xy #.cpu().numpy()

    text_size = 18
    canvas = 255 * np.ones(
        (cell_map.shape[0], canvas_size + text_size, canvas_size, 3), np.uint8)
    # Coordinate transform + rounding to nearest int
    tr = lambda c: int(0.5 + ((0.5 + c) * canvas_size / 30))

    for b_id in range(cell_map.shape[0]):
        curr_cell_map = cell_map[b_id]
        curr_canv = canvas[b_id]
        agent_x, agent_y, agent_tb, _  = agent_cfg[b_id].tolist()
        target_x, target_y = target_xy[b_id].tolist()

        direction_map = {
            0: ( 1, 0),
            1: ( 0,-1),
            2: (-1, 0),
            3: ( 0, 1),
        }

        # Drawing objects with rectangles
        for ids in zip(*np.where(curr_cell_map == 1)):
            _x, _y = ids
            if _x == agent_x and _y == agent_y:
                continue # Do not draw over agent occupied cell
            cv2.rectangle(curr_canv,
                          pt1 = (tr(_x), tr(_y)),
                          pt2 = (tr(_x + 1), tr(_y + 1)),
                          color=(0,0,0),
                          thickness=-1)

        # Drawing agent with circle + line
        AGENT_COLOR = (0, 0, 255)
        if done_flag is not None:
            if done_flag[b_id]:
                AGENT_COLOR = (0, 255, 0)
        cv2.circle(curr_canv,
                   center=(tr(agent_x), tr(agent_y)),
                   radius=int(tr(1)/2),
                   color=AGENT_COLOR,
                   thickness=max(int(tr(1)/4),2))
        dr_x, dr_y = direction_map[int(agent_tb)]
        dr_x = 2.0*dr_x
        dr_y = 2.0*dr_y
        cv2.line(curr_canv,
                 pt1=(tr(agent_x), tr(agent_y)),
                 pt2=(tr(agent_x + dr_x), tr(agent_y + dr_y)),
                 color=(0, 0, 255),
                 thickness=max(int(tr(1)/4),1))

        # Draw target with circle
        cv2.circle(curr_canv,
                   center=(tr(target_x), tr(target_y)),
                   radius=int(tr(1)),
                   color=(255, 0, 0),
                   thickness=max(int(tr(1)/3),2))

        # Draw text with agent X, Y, body orientation
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.line(curr_canv,
                 pt1=(0, canvas_size),
                 pt2=(canvas_size, canvas_size),
                 color=(0,0,0),
                 thickness=1)
        cv2.putText(curr_canv,
                   text='({},{}), ({})'.format(
                        int(agent_x), int(agent_y), int(agent_tb)),
                   org=(0, canvas_size + text_size - 2),
                   fontFace=font,
                   fontScale=1,
                   color=(0,0,0),
                   thickness=2,
                   lineType=cv2.LINE_AA)
        canvas[b_id] = curr_canv

    canvas = canvas.transpose([0, 3, 1, 2]).astype('float')*(1.0 / 255)
    canvas = torch.from_numpy(canvas).float()
    return canvas

def draw_trajectory(trajectory, canvas_size):
    # agent_cfg = agent_cfg #.cpu().numpy()
    # target_xy = target_xy #.cpu().numpy()

    # text_size = 18
    # canvas = 255 * np.ones(
    #     (cell_map.shape[0], canvas_size + text_size, canvas_size, 3), np.uint8)]

    done = np.stack(trajectory['done'], 0)
    agent_xy = np.stack(trajectory['agent_xy'], 0)
    targets = trajectory['target']

    color_map = {
        0 : (190, 190, 190), #gray
        1 : (255, 0, 0), #red
        2 : (0, 0, 255), #blue
        3 : (0, 100, 0), #green
        4 : (165, 42, 42), #briwb
        5 : (160, 32, 240), #purple
        6 : (0, 255, 255), #cyan
        7 : (255, 255, 0), #yellow
    }
    color_arr = np.array([list(item) for item in color_map.values()])

    cmap = trajectory['cmap']
    cmap = cmap.reshape(*cmap.shape[:3], 3, 8, 2, 2)
    cell_map = cmap.sum((3, 5, 6)) #.argmax(-1)
    grid = 255 * np.ones((cell_map.shape[0], 30, 30, 3), np.uint8)

    for color_idx in range(8):
        grid[np.where(cell_map[:, :, :, color_idx] == 1)] = \
            color_map[color_idx]

    # Coordinate transform + rounding to nearest int
    tr = lambda c: int(0.5 + ((0.5 + c) * canvas_size / 30))

    for b_id in range(cell_map.shape[0]):
        # canvas = 255 * np.ones(
        #     (cell_map.shape[0], canvas_size, canvas_size, 3), np.uint8)
        can = cv2.resize(grid[b_id],
            dsize=(canvas_size, canvas_size), interpolation=cv2.INTER_NEAREST)


        curr_cell_map = cell_map[b_id]
        curr_canv = can
        curr_xy = agent_xy[:, b_id]
        curr_done = done[:, b_id]

        _xy = curr_xy[0]
        agent_x, agent_y = _xy

        # Drawing agent with circle + line
        AGENT_COLOR = (0, 100, 0)
        END_COLOR = (218, 112, 214)
        # if done_flag is not None:
        #     if done_flag[b_id]:
        #         AGENT_COLOR = (0, 255, 0)
        cv2.circle(curr_canv,
                   center=(tr(agent_x), tr(agent_y)),
                   radius=int(tr(1)/2),
                   color=AGENT_COLOR,
                   thickness=-max(int(tr(1)/4),2))

        prev_xy = curr_xy[0]
        prev_x, prev_y = prev_xy

        import pdb; pdb.set_trace()

        for t_id in range(1, curr_xy.shape[0]):
            # _xy = curr_xy[t_id]
            agent_x, agent_y = curr_xy[t_id]

            # Drawing agent with circle + line
            # if done_flag is not None:
            #     if done_flag[b_id]:
            #         AGENT_COLOR = (0, 255, 0)
            cv2.line(curr_canv,
                     pt1=(tr(agent_x), tr(agent_y)),
                     pt2=(tr(prev_x), tr(prev_y)),
                     # pt2=(tr(agent_x + dr_x), tr(agent_y + dr_y)),
                     color=(0, 0, 255),
                     thickness=max(int(tr(1)/5),1))

            if t_id == curr_xy.shape[0] - 1 or curr_done == 1:
                cv2.circle(curr_canv,
                           center=(tr(agent_x), tr(agent_y)),
                           radius=int(tr(1)/2),
                           color=END_COLOR,
                           thickness=-max(int(tr(1)/4),2))
                break
            prev_x = agent_x
            prev_y = agent_y

        cv2.imwrite("tmp1.png", curr_canv)
        import pdb; pdb.set_trace()

        # agent_x, agent_y, agent_tb, _  = agent_cfg[b_id].tolist()
        # target_x, target_y = target_xy[b_id].tolist()

        direction_map = {
            0: ( 1, 0),
            1: ( 0,-1),
            2: (-1, 0),
            3: ( 0, 1),
        }

        # Drawing objects with rectangles
        for ids in zip(*np.where(curr_cell_map == 1)):
            _x, _y = ids
            if _x == agent_x and _y == agent_y:
                continue # Do not draw over agent occupied cell
            cv2.rectangle(curr_canv,
                          pt1 = (tr(_x), tr(_y)),
                          pt2 = (tr(_x + 1), tr(_y + 1)),
                          color=(0,0,0),
                          thickness=-1)

        # Drawing agent with circle + line
        AGENT_COLOR = (0, 0, 255)
        if done_flag is not None:
            if done_flag[b_id]:
                AGENT_COLOR = (0, 255, 0)
        cv2.circle(curr_canv,
                   center=(tr(agent_x), tr(agent_y)),
                   radius=int(tr(1)/2),
                   color=AGENT_COLOR,
                   thickness=max(int(tr(1)/4),2))
        dr_x, dr_y = direction_map[int(agent_tb)]
        dr_x = 2.0*dr_x
        dr_y = 2.0*dr_y
        cv2.line(curr_canv,
                 pt1=(tr(agent_x), tr(agent_y)),
                 pt2=(tr(agent_x + dr_x), tr(agent_y + dr_y)),
                 color=(0, 0, 255),
                 thickness=max(int(tr(1)/4),1))

        # Draw target with circle
        cv2.circle(curr_canv,
                   center=(tr(target_x), tr(target_y)),
                   radius=int(tr(1)),
                   color=(255, 0, 0),
                   thickness=max(int(tr(1)/3),2))

        # Draw text with agent X, Y, body orientation
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.line(curr_canv,
                 pt1=(0, canvas_size),
                 pt2=(canvas_size, canvas_size),
                 color=(0,0,0),
                 thickness=1)
        cv2.putText(curr_canv,
                   text='({},{}), ({})'.format(
                        int(agent_x), int(agent_y), int(agent_tb)),
                   org=(0, canvas_size + text_size - 2),
                   fontFace=font,
                   fontScale=1,
                   color=(0,0,0),
                   thickness=2,
                   lineType=cv2.LINE_AA)
        canvas[b_id] = curr_canv

    canvas = canvas.transpose([0, 3, 1, 2]).astype('float')*(1.0 / 255)
    canvas = torch.from_numpy(canvas).float()
    return canvas


def concat_horizontal(img1, img2):
    if img1.shape[2] != img2.shape[2]:
        if img1.shape[2] < img2.shape[2]:
            img_small, img_big = img1, img2
            order_reverse = False
        else:
            img_small, img_big = img2, img1
            order_reverse = True
        scale = (1.0 * img_big.shape[2]) / img_small.shape[2]
        with torch.no_grad():
            img_small = F.interpolate(
                img_small, scale_factor=scale, mode='bilinear')
    else:
        order_reverse = False
        img_small, img_big = img1, img2

    if order_reverse:
        return torch.cat([img_big, img_small], 3)
    else:
        return torch.cat([img_small, img_big], 3)


def get_video_frame(envs,num_agents):

    agent_pov_img = [env.get_processed_pov_img() for env in envs]

    curr_agent_cfg = np.concatenate(
            [env.get_agent_cfg() for env in envs],0)
    cell_map = np.concatenate(
            [env.get_cell_occ_map() for env in envs],0)
    target_xy = np.concatenate(
            [env.targets[1] for env in envs],0)

    top_down_map = draw_map(
            curr_agent_cfg, target_xy, cell_map, done_flag=None)


    agent_pov_img = np.concatenate(agent_pov_img,0)
    agent_pov_img = torch.from_numpy(agent_pov_img).float()

    agent_pov_img = torch.cat(agent_pov_img.split([3,3,3],1),3)

    flat_frame = concat_horizontal(agent_pov_img, top_down_map)

    frame_shape = flat_frame.shape[1:]

    # Return Envs x Agents x Frame x Channel x H x W

    frame = flat_frame.view((-1,num_agents, 1,) + frame_shape)

    return frame
