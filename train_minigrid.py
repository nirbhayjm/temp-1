import time
import argparse
import numpy as np
import os
import json
import random
from collections import OrderedDict

import torch
import torch.optim as optim

import gym
from gym.envs.registration import register
# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from utilities.subproc_vec_env import  SubprocVecEnvCustom as SubprocVecEnv

from envs.mdp_envs.two_goals_behind_wall import TwoGoalsBehindWall
from envs.mdp_envs.random_two import RandomTwoGoals
from envs.mdp_envs.one_corner import OneCornerGoal
from envs.mdp_envs.corridor import NarrowCorridor
from envs.mdp_envs.multiroom_mdp import MultiroomMDP
from envs.mdp_envs.crossing_mdp import CrossingMDP
from envs.mdp_envs.pillar_grid import PillarGrid
from envs.pomdp_envs.multiroom import MultiroomWrapper
from envs.pomdp_envs.crossing import CrossingEnvWrapper
from envs.mdp_envs.mountain_car import MountainCarEnvCustom
from envs.mdp_envs.acrobat import AcrobatEnvCustom

from envs.bonus_reward import BonusReward

import algo
from policy.models import Policy
from policy.hierarchical_models import (TrajectoryEncoder,
                                        OptionsDecoder,
                                        DummyModule,
                                        IBPolicy,
                                        IBEncoder,
                                        IBSupervisedPolicy,
                                        IBSupervisedEncoder,
                                        IBSupervisedTDPolicy,
                                        OptionsPolicy)
# from flow_models import PlanarOptionsDecoder, IAFOptionsDecoder
from storage import RolloutStorage, DictObs
from td_storage import RolloutStorageTD

from arguments import get_args
from trainer import Trainer
from trainer_td import TrainerTD
import utilities.utilities as utils
from utilities.utilities import make_env
# import utilities.eval_utils as eval_utils

# [NOTE]: Gym specific functions, move to a separate helper file
# once things start working
# def make_env(env_id, seed, rank, config_seed=None):
#     def _thunk():
#         env = gym.make(env_id)
#         env.seed(seed + rank)
#         if config_seed is not None:
#             env.seed_config(config_seed + rank)
#         return env
#
#     return _thunk

SESSION_PARAMS = ['viz_dir', 'log_dir', 'save_dir', 'save_sub_dir',
    'ac_start_from', 'num_agents', 'num_processes', 'ppo_epoch',
    'num-mini-batch', 'dataloader_workers', 'no_cuda', 'visdom_env_name',
    'vis', 'server', 'port', 'log_interval', 'save_interval', 'identifier',
    'val_interval', 'skip_eval', 'vid_interval', 'vid_envs', 'vid_agents',
    'vis_heatmap', 'num_episodes', 'hier_mode', 'infobot_beta_min',
    'infobot_beta_curriculum', 'reward_scale', 'infobot_kl_start',
    'infobot_kl_growth', 'reset_prob', 'num_steps', 'num_option_steps',
    'hr_model_kl_coeff', 'heatmap_interval']
SESSION_PARAMS += ['time_id', 'gpus', 'jobid', 'slurm_node',
    'slurm_job_user', 'slurm_task_pid', 'slurm_job_partition',
    'output_log_path']


if __name__ == '__main__':
    args = get_args()

    if args.ac_start_from:
        to_load_params = ['omega_dim_current']

        # utils.load_from_ckpt(actor_critic, args.ac_start_from)
        args, start_iter = utils.load_args(
            args=args,
            session_params=SESSION_PARAMS,
            to_load_params=to_load_params,
            ckpt_path=args.ac_start_from)
        if args.hier_mode == 'transfer' or args.hier_mode == 'bonus':
            start_iter = 0

    else:
        start_iter = 0

    if args.hier_mode == 'bonus':
        if args.bonus_heuristic_beta <= args.bonus_beta:
            args.bonus_heuristic_beta = -1

    b_args = {}
    if args.
        # to_load_params = []
        # # utils.load_from_ckpt(actor_critic, args.ac_start_from)
        # b_args, _ = utils.load_args(
        #     args=args,
        #     session_params=SESSION_PARAMS,
        #     to_load_params=to_load_params,
        #     ckpt_path=args.
        z_enc_ckpt = torch.load(args.
        if 'omega_dim_current' not in z_enc_ckpt['params']:
            print("WARNING!: Using default 2 options for checkpoint")
            z_enc_ckpt['params']['omega_dim_current'] = 2
        b_args = argparse.Namespace(**z_enc_ckpt['params'])


    # 'feat_sim' not supported yet
    if args.reward_type == 'feat_sim':
        assert args.model == 'cond'
        raise NotImplementedError
    # if args.base_model == 'full_state':
    #     raise NotImplementedError

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.time_id = time.strftime("%m_%d_%H:%M")

    if args.no_cuda:
        print("[WARNING] Using CPU!")
    else:
        try:
            args.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
            args.jobid = os.environ['SLURM_JOBID']
            args.slurm_node = os.environ['SLURM_NODELIST']
            args.slurm_job_user = os.environ['SLURM_JOB_USER']
            args.slurm_task_pid = os.environ['SLURM_TASK_PID']
            args.slurm_job_partition = os.environ['SLURM_JOB_PARTITION']

            # Log the SLURM_JOBID in the args.log-dir
            dir_name = os.path.join(args.log_dir, args.identifier)
            os.makedirs(dir_name, exist_ok=True)
            fpath = os.path.join(dir_name, args.jobid + '_' \
                + args.visdom_env_name + '_' + args.time_id +'.json')
            print(fpath)
            with open(fpath, 'w') as f:
                json.dump(vars(args), f)
            # if len(args.gpus) == 2:
            #     args.gpus = [args.gpus[1]]
        except KeyError:
            print("GPU not found!")
            exit()

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    '''Hierarchical model specific arguments'''
    assert args.target_embed_type == 'k-hot'
    '''END'''

    #####################
    # ENVIRONMENT SETUP #
    #####################
    if args.grid_obs_type == 'xy':
        use_grid_in_state = False
    elif args.grid_obs_type == 'grid':
        use_grid_in_state = True
    else:
        raise ValueError

    pomdp_env_names = ['multiroom', 'crossing', 'empty-pomdp',
        'two-room-corridor-pomdp', 'four-rooms-pomdp', 'maze-pomdp']
    args.is_pomdp_env = False
    if args.env_name in pomdp_env_names:
        args.is_pomdp_env = True

    if args.env_name == 'multiroom':
        use_heuristic_ds = args.bonus_heuristic_beta > args.bonus_beta

        if (args.hier_mode == 'bonus' or args.hier_mode == 'infobot-supervised') \
        and args.use_td:
            max_steps = args.multiroom_num_rooms * 20
            reset_on_done = True
        else:
            max_steps = args.num_steps
            reset_on_done = False

        train_kwargs = {
            'size': args.env_grid_size,
            'spawn': args.spawn_curriculum,
            'num_rooms': args.multiroom_num_rooms,
            'max_steps': max_steps,
            'term_prob': args.term_prob,
            'reset_prob': args.reset_prob,
            'agent_view_size': args.agent_view_size,
            'obs_win_size': args.obs_win_size,
            'reward_scale': args.reward_scale,
            'end_on_goal': args.end_on_goal,
            'use_heuristic_ds': use_heuristic_ds,
            'obs_alpha': 0.0,
            'static_env_grid': args.static_env_grid,
            'config_seed': args.config_seed,
            'doors_open': args.multiroom_doors_open,
            'reset_on_done': reset_on_done,
            'seed': args.seed,
        }

    elif args.env_name == 'multiroom-mdp':
        train_kwargs = {
            'max_steps': args.num_steps,
            'term_prob': args.term_prob,
            'reset_prob': args.reset_prob,
            # 'obs_alpha': 0.001,
            'obs_alpha': 0.0,
            'seed': args.seed,
        }

    elif args.env_name == 'crossing' or 'pomdp' in args.env_name:
        if args.env_name == 'empty-pomdp':
            grid_type = 'empty'
        elif args.env_name == 'two-room-corridor-pomdp':
            grid_type = 'two-room-corridor'
        elif args.env_name == 'four-rooms-pomdp':
            grid_type = 'four-rooms'
        elif args.env_name == 'maze-pomdp':
            grid_type = 'maze'
        elif args.env_name == 'crossing':
            grid_type = 'crossing'
        else:
            raise ValueError("Unknown env_name: {}".format(args.env_name))

        transfer_mode = False
        max_steps = args.num_steps
        reset_on_done = False
        if args.hier_mode == 'bonus':
            transfer_mode = True
            max_steps = 4 * args.env_grid_size
            reset_on_done = True

        elif args.hier_mode == 'infobot-supervised' and args.use_td:
            reset_on_done = True
            max_steps = 4 * args.env_grid_size

        train_kwargs = {
            'size': args.env_grid_size,
            'obstacle_type': 'wall',
            'grid_type': grid_type,
            'num_crossings': 2,
            'obs_win_size': args.obs_win_size,
            'agent_view_size': args.agent_view_size,
            'corridor_len': args.corridor_len,
            'complexity': args.maze_complexity,
            'density': args.maze_density,
            'max_steps': max_steps,
            'term_prob': args.term_prob,
            'reset_prob': args.reset_prob,
            'reward_scale': args.reward_scale,
            'static_env_grid': args.static_env_grid,
            'reset_on_done': reset_on_done,
            'transfer_mode': transfer_mode,
            'spawn': args.spawn_curriculum,
            'end_on_goal': args.end_on_goal,
            'perturb_prob': args.perturb_prob,
            'randomize_goal_pos': args.randomize_goal_pos,
            # 'obs_alpha': 0.001,
            'obs_alpha': 0.0,
            'config_seed': args.config_seed,
            'seed': args.seed,
        }

    elif args.env_name == 'mountain-car':
        train_kwargs = {
            'max_steps': args.num_steps,
            'reward_scale': args.reward_scale,
            'spawn': args.spawn_curriculum,
            'reset_prob': args.reset_prob,
            # 'perturb_prob': args.perturb_prob,
            # 'term_prob': args.term_prob,
            # 'obs_alpha': 0.001,
            # 'obs_alpha': 0.0,
            # 'config_seed': args.config_seed,
            'reset_on_done': False,
            'end_on_goal': args.end_on_goal,
            'seed': args.seed,
        }

    elif args.env_name == 'acrobat':
        train_kwargs = {
            'max_steps': args.num_steps,
            'reward_scale': args.reward_scale,
            # 'spawn': args.spawn_curriculum,
            'reset_prob': args.reset_prob,
            'reset_on_done': False,
            'end_on_goal': args.end_on_goal,
            'seed': args.seed,
        }

    else:
        # if args.env_name == 'multiroom-mdp':
        #     size_kwargs = {
        #         'maxRoomSize': 25,
        #         'maxNumRooms': 5,
        #         'minNumRooms': 5,
        #     }
        if args.env_name == 'corridor':
            # _SIZE = 4
            size_kwargs = {'size': 4}
        elif args.env_name == 'crossing-mdp':
            size_kwargs = {
                'size': args.env_grid_size,
                'num_crossings': 3,
                'obstacle_type': 'wall',
            }
        elif args.env_name == 'maze':
            # Maze generator expects odd sized grid. If even size is passed, it is automatically increased by 1.
            # To make this even to odd conversion more transparent, doing it here.

            # assert args.num_steps >= 2 * args.env_grid_size, \
            #     "Make 'num_steps' at least 2 * 'env_grid_size"

            if args.env_grid_size % 2 == 0:
                args.env_grid_size += 1
                size_kwargs = {'size': args.env_grid_size}
            else:
                size_kwargs = {'size': args.env_grid_size}

            # size_kwargs['config_seed'] = args.config_seed
            size_kwargs['static_grid'] = args.static_env_grid
            size_kwargs['complexity'] = args.maze_complexity
            size_kwargs['density'] = args.maze_density

        elif args.env_name == 'two-room-corridor':
            size_kwargs = {
                'size' : args.env_grid_size,
                'corridor_len' : args.corridor_len,
                'agent_start_room' : 'left',
            }
        else:
            size_kwargs = {'size': args.env_grid_size}


        train_kwargs = {
            # 'size': _SIZE,
            **size_kwargs,
            'max_steps': args.num_steps,
            'use_grid_in_state': use_grid_in_state,
            'normalize_agent_pos': True,
            'reward_scale': args.reward_scale,
            'spawn': args.spawn_curriculum,
            'reset_prob': args.reset_prob,
            'perturb_prob': args.perturb_prob,
            # 'transfer_mode': transfer_mode,
            'term_prob': args.term_prob,
            # 'obs_alpha': 0.001,
            'obs_alpha': 0.0,
            'config_seed': args.config_seed,
            'seed': args.seed,
        }

    vis_kwargs = train_kwargs.copy()
    vis_kwargs['render_rgb'] = True
    val_kwargs = vis_kwargs.copy()

    # print("[WARN]: Overwriting args.num_steps!")
    # args.num_steps = 30 * (train_kwargs['size'] ** 2)
    # print("Num steps: {}".format(args.num_steps))

    ##############################
    # Validation metric logging  #
    ##############################

    if args.observability in ('partial-single', 'partial-triplet'):
        raise NotImplementedError
        print("Viewpoint loading ON")
        get_viewpoints = True
    elif args.observability == 'full':
        print("Viewpoint loading OFF")
        get_viewpoints = False
    else:
        raise NotImplementedError

    continuous_state_space = False
    dir_root = 'mdp_envs'
    if args.env_name == 'randomtwo':
        env_name = 'RandomTwoGoals'
        env_root = 'random_two:RandomTwoGoals'
    elif args.env_name == 'twogoals':
        env_name = 'TwoGoalsBehindWall'
        env_root = 'two_goals_behind_wall:TwoGoalsBehindWall'
    elif args.env_name == 'onecorner':
        env_name = 'OneCornerGoal'
        env_root = 'one_corner:OneCornerGoal'
    elif args.env_name == 'corridor':
        env_name = 'NarrowCorridor'
        env_root = 'corridor:NarrowCorridor'
    elif args.env_name == 'multiroom-mdp':
        env_name = 'MultiroomMDP'
        env_root = 'multiroom_mdp:MultiroomMDP'
    elif args.env_name == 'mountain-car':
        continuous_state_space = True
        env_name = 'MountainCarEnvCustom'
        env_root = 'mountain_car:MountainCarEnvCustom'
    elif args.env_name == 'acrobat':
        continuous_state_space = True
        env_name = 'AcrobatEnvCustom'
        env_root = 'acrobat:AcrobatEnvCustom'
    elif args.env_name == 'multiroom':
        dir_root = 'pomdp_envs'
        env_name = 'MultiroomWrapper'
        env_root = 'multiroom:MultiroomWrapper'
    elif args.env_name == 'two-room-corridor':
        env_name = 'TwoRoomCorridor'
        env_root = 'two_room_corridor:TwoRoomCorridor'
    elif args.env_name == 'four-rooms':
        env_name = 'FourRooms'
        env_root = 'four_rooms:FourRooms'
    elif args.env_name == 'maze':
        env_name = 'Maze'
        env_root = 'maze:Maze'
    elif args.env_name == 'crossing-mdp':
        env_name = 'CrossingMDP'
        env_root = 'crossing_mdp:CrossingMDP'
    elif args.env_name == 'pillar-grid':
        env_name = 'PillarGrid'
        env_root = 'pillar_grid:PillarGrid'
    elif args.env_name == 'crossing' or 'pomdp' in args.env_name:
        dir_root = 'pomdp_envs'
        env_name = 'CrossingEnvWrapper'
        env_root = 'crossing:CrossingEnvWrapper'
    else:
        raise ValueError

    register(
        id='{}-v0'.format(env_name),
        entry_point=\
            'envs.{}.{}'.format(dir_root, env_root),
        kwargs=train_kwargs,
    )

    register(
        id='{}-val-v0'.format(env_name),
        entry_point=\
            'envs.{}.{}'.format(dir_root, env_root),
        kwargs=val_kwargs,
    )

    register(
        id='{}-vis-v0'.format(env_name),
        entry_point=\
            'envs.{}.{}'.format(dir_root, env_root),
        kwargs=vis_kwargs,
    )

    if args.static_env_grid:
        train_envs = [
            make_env('{}-v0'.format(env_name),args.seed,i) \
            for i in range(args.num_processes)
        ]
    else:
        train_envs = [
            make_env('{}-v0'.format(env_name),
                args.seed,i,config_seed=args.config_seed) \
            for i in range(args.num_processes)
        ]

    val_envs = [
        make_env('{}-val-v0'.format(env_name),
            args.seed,i,config_seed=args.config_seed) \
        for i in range(args.num_processes)
    ]

    vis_env = make_env('{}-vis-v0'.format(env_name), args.seed, 0)()

    if args.env_name not in ['mountain-car', 'acrobat']:
        bonus_reward = BonusReward(env=vis_env, beta=args.bonus_beta)
    else:
        bonus_reward = None

    # '''DEBUG'''
    # from utilities.visualize import VisdomLogger
    # viz = VisdomLogger(
    #     env_name=args.visdom_env_name,
    #     server=args.server,
    #     port=args.port)
    #
    # def _random_img(_name="img"):
    #     vis_obs, vis_info = vis_env.reset()
    #     assert 'rgb_grid' in vis_info
    #     env_rgb_img = vis_info['rgb_grid'].transpose([2, 0, 1])
    #     viz.image(np.flip(env_rgb_img, 1), _name)
    # import pdb; pdb.set_trace()
    # '''DEBUG'''

    # '''DEBUG'''
    # print("DEBUGGING MODE ON")
    # env = vis_env
    # obs = env.reset()
    # # obs = env.step(env.actions.right); print(env); print(obs[1], obs[2])
    # import pdb; pdb.set_trace()
    # print(env)
    # tmp_obs2 = env.step(env.actions.forward)
    # print(env)
    # # print(env.step(env.actions.forward))
    # print(env.step(env.actions.done))
    # print(env)
    # import pdb; pdb.set_trace()
    # '''DEBUG'''

    train_envs = SubprocVecEnv(train_envs)
    val_envs = SubprocVecEnv(val_envs)

    action_count = train_envs.action_space.n

    if not get_viewpoints:
        args.dataloader_workers = 1

    #######################
    # ACTOR-CRITIC Model  #
    #######################

    obs_spaces = train_envs.observation_space.spaces

    if args.hier_mode == 'vic':
        obs_spaces.pop('goal_vector', None)

    rollout_obs_spaces = obs_spaces.copy()

    if args.is_pomdp_env:
        pos_space = obs_spaces.pop('pos')


    base_obs_spaces = obs_spaces.copy()
    if args.model == 'cond':
        pass
        opt_dec_obs_spaces = None
        z_encoder_obs_spaces = obs_spaces.copy()
        z_encoder_obs_spaces.pop('mission', None)
        if b_args.hier_mode == 'vic':
            z_encoder_obs_spaces.pop('goal_vector', None)
            z_encoder_obs_spaces.update({
                'omega': gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(b_args.omega_option_dims,),
                    dtype='float',
                )
            })

            z_enc_policy_obs_spaces = z_encoder_obs_spaces.copy()
            z_enc_policy_obs_spaces.pop('omega')
            z_enc_policy_obs_spaces.update({
                'z_latent': gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(b_args.z_latent_dims,),
                    dtype='float',
                )
            })
        elif b_args.hier_mode == 'infobot-supervised':
            z_enc_policy_obs_spaces = z_encoder_obs_spaces.copy()
            # z_enc_policy_obs_spaces.pop('goal_vector')
            # z_enc_policy_obs_spaces.update({
            #     'z_latent': gym.spaces.Box(
            #         low=0,
            #         high=1,
            #         shape=(b_args.z_latent_dims,),
            #         dtype='float',
            #     )
            # })
        else:
            raise ValueError

    else:
        if args.hier_mode == 'infobot-supervised':
            # assert 'goal_vector' in obs_spaces
            ib_policy_obs_spaces = obs_spaces.copy()
            ib_policy_obs_spaces.pop('mission')

            # base_obs_spaces.pop('goal_vector')
            base_obs_spaces.pop('mission')
            # base_obs_spaces.update({
            #     'z_latent': gym.spaces.Box(
            #         low=0,
            #         high=1,
            #         shape=(args.z_latent_dims,),
            #         dtype='float',
            #     )
            # })

        else:
            if args.use_infobot:
                base_obs_spaces.update({
                    'z_latent': gym.spaces.Box(
                        low=0,
                        high=1,
                        shape=(args.z_latent_dims,),
                        dtype='float',
                    )
                })
                base_obs_spaces.pop('mission')
                ib_policy_obs_spaces = obs_spaces.copy()
                ib_policy_obs_spaces.pop('mission', None)
                ib_policy_obs_spaces.update({
                    'omega': gym.spaces.Box(
                        low=0,
                        high=1,
                        shape=(args.omega_option_dims,),
                        dtype='float',
                    )
                })
            else:
                base_obs_spaces.update({
                    'omega': gym.spaces.Box(
                        low=0,
                        high=1,
                        shape=(args.omega_option_dims,),
                        dtype='float',
                    )
                })

        opt_dec_obs_spaces = obs_spaces.copy()
        opt_dec_obs_spaces.pop('mission', None)

    mlp_base_kwargs = {
        'obs_spaces': base_obs_spaces,
        'recurrent': bool(args.recurrent_policy),
        'hidden_size': args.hidden_size,
        'critic_detach': args.critic_detach,
    }

    # Policy
    ib_policy_kwargs = {}
    if args.use_infobot and args.model != 'cond':

        if args.hier_mode == 'infobot-supervised':
            assert args.recurrent_encoder == args.recurrent_policy
            assert bool(args.recurrent_policy)
            mlp_base_kwargs['use_critic'] = False
            ib_policy_kwargs = {
                'action_dims' : action_count,
                'observability': args.observability,
                'obs_spaces': obs_spaces,
                'hidden_size': args.hidden_size,
                'latent_space': 'gaussian',
                'z_dims': args.z_latent_dims,
                'base_model': args.base_model,
                'z_std_clip_max': args.z_std_clip_max,
                'base_kwargs': {
                    'obs_spaces': mlp_base_kwargs,
                    'recurrent': bool(args.recurrent_encoder),
                    'hidden_size': args.hidden_size,
                    'use_critic': False,
                },
                'policy_base_kwargs': mlp_base_kwargs,
            }
            if args.use_td:
                actor_critic = IBSupervisedTDPolicy(**ib_policy_kwargs)
            else:
                actor_critic = IBSupervisedPolicy(**ib_policy_kwargs)
        else:
            ib_policy_kwargs = {
                'action_dims' : action_count,
                'observability': args.observability,
                'obs_spaces': obs_spaces,
                'hidden_size': args.hidden_size,
                'latent_space': 'gaussian',
                'z_dims': args.z_latent_dims,
                'base_model': args.base_model,
                'z_std_clip_max': args.z_std_clip_max,
                'base_kwargs': {
                    'obs_spaces': ib_policy_obs_spaces,
                    'recurrent': bool(args.recurrent_encoder),
                    'hidden_size': args.hidden_size,
                    'use_critic': False,
                },
                'policy_base_kwargs': mlp_base_kwargs,
            }
            actor_critic = IBPolicy(**ib_policy_kwargs)
        # encoder_recurrent_hidden_state_size = \
        #     actor_critic.encoder_recurrent_hidden_state_size
    else:
        policy_kwargs = {
            'observability' : args.observability,
            'action_dims' : action_count,
            'base_model': args.base_model,
            'base_kwargs' : mlp_base_kwargs,
        }
        actor_critic = Policy(**policy_kwargs)
        # encoder_recurrent_hidden_state_size = 1

    z_encoder = None
    if args.
        if b_args.use_infobot:
            if args.bonus_type == 'kl':
                if b_args.hier_mode == 'infobot-supervised':
                    z_enc_policy_kwargs = {
                        'action_dims': action_count,
                        'observability': b_args.observability,
                        'obs_spaces': obs_spaces,
                        'hidden_size': b_args.hidden_size,
                        'latent_space': 'gaussian',
                        'z_dims': b_args.z_latent_dims,
                        'z_std_clip_max': args.z_std_clip_max,
                        'base_model': b_args.base_model,
                        'base_kwargs': {
                            'obs_spaces': z_encoder_obs_spaces,
                            'recurrent': bool(b_args.recurrent_encoder),
                            'hidden_size': b_args.hidden_size,
                            'use_critic': False,
                        },
                        'policy_base_kwargs': {
                            'obs_spaces': z_enc_policy_obs_spaces,
                            'recurrent': bool(b_args.recurrent_policy),
                            'hidden_size': b_args.hidden_size,
                            'use_critic': False,
                        },
                    }
                    if hasattr(b_args, 'use_td') and b_args.use_td:
                        z_encoder = IBSupervisedTDPolicy(**z_enc_policy_kwargs)
                        z_enc_mdict = z_enc_ckpt['model']
                    else:
                        z_encoder = IBSupervisedPolicy(**z_enc_policy_kwargs)
                        z_enc_mdict = z_enc_ckpt['model']
                        # loaded_ib_policy_mdict = z_enc_ckpt['model']
                        # z_enc_mdict = OrderedDict()
                        # for key in loaded_ib_policy_mdict.keys():
                        #     if key.startswith('ib_encoder'):
                        #         new_key = key.split('ib_encoder.')[-1]
                        #         z_enc_mdict[new_key] = loaded_ib_policy_mdict[key]

                else:
                    z_encoder_kwargs = {
                        'observability': b_args.observability,
                        'obs_spaces': obs_spaces,
                        'hidden_size': b_args.hidden_size,
                        'latent_space': 'gaussian',
                        'z_dims': b_args.z_latent_dims,
                        'z_std_clip_max': args.z_std_clip_max,
                        'base_model': b_args.base_model,
                        'base_kwargs': {
                            'obs_spaces': z_encoder_obs_spaces,
                            'recurrent': bool(b_args.recurrent_encoder),
                            'hidden_size': b_args.hidden_size,
                            'use_critic': False,
                        },
                    }
                    z_encoder = IBEncoder(**z_encoder_kwargs)
                    loaded_ib_policy_mdict = z_enc_ckpt['model']
                    z_enc_mdict = OrderedDict()
                    for key in loaded_ib_policy_mdict.keys():
                        if key.startswith('ib_encoder'):
                            new_key = key.split('ib_encoder.')[-1]
                            z_enc_mdict[new_key] = loaded_ib_policy_mdict[key]

            else:
                z_enc_policy_kwargs = {
                    'action_dims': action_count,
                    'observability': b_args.observability,
                    'obs_spaces': obs_spaces,
                    'hidden_size': b_args.hidden_size,
                    'latent_space': 'gaussian',
                    'z_dims': b_args.z_latent_dims,
                    'z_std_clip_max': args.z_std_clip_max,
                    'base_model': b_args.base_model,
                    'base_kwargs': {
                        'obs_spaces': z_encoder_obs_spaces,
                        'recurrent': bool(b_args.recurrent_encoder),
                        'hidden_size': b_args.hidden_size,
                        'use_critic': False,
                    },
                    'policy_base_kwargs': {
                        'obs_spaces': z_enc_policy_obs_spaces,
                        'recurrent': bool(b_args.recurrent_policy),
                        'hidden_size': b_args.hidden_size,
                        'critic_detach': True,
                    },
                }
                z_encoder = IBPolicy(**z_enc_policy_kwargs)
                z_enc_mdict = z_enc_ckpt['model']
                # z_enc_mdict = OrderedDict()
                # for key in loaded_ib_policy_mdict.keys():
                #     # if key.startswith('ib_encoder'):
                #     #     new_key = key.split('ib_encoder.')[-1]
                #     z_enc_mdict[key] = loaded_ib_policy_mdict[key]
        else:
            raise NotImplementedError

        arg_dict = vars(args)
        b_arg_dict = vars(b_args)
        for key in arg_dict.keys():
            if key in SESSION_PARAMS:
                continue
            if key not in b_arg_dict:
                print(f"Key [{key}] not present in checkpoint.")
                continue
            if arg_dict[key] != b_arg_dict[key]:
                print(f"Key [{key}] mismatch arg={arg_dict[key]}; b_arg={b_arg_dict[key]}")

        if args.bonus_type != 'count':
            if args.bonus_type == 'kl':
                assert b_args.use_infobot == 1
                assert b_args.z_stochastic == 1

            elif args.bonus_type == 'kl-pi':
                pass

            assert b_args.agent_view_size == args.agent_view_size,\
                f"Agent view size mismatch in enc ckpt {b_args.agent_view_size} "\
                f"and current model {args.agent_view_size}"

            z_encoder.load_state_dict(z_enc_mdict)

        if b_args.hier_mode == 'infobot-supervised':
            if hasattr(b_args, 'use_td') and b_args.use_td:
                pass
            else:
                z_encoder = z_encoder.ib_encoder

    options_policy = None
    if args.hier_mode == 'transfer':
        options_policy_kwargs = {
            'observability' : args.observability,
            'option_dims' : args.omega_option_dims,
            'option_space': args.option_space,
            'hidden_size': args.hidden_size,
            'base_model': args.base_model,
            'base_kwargs' : {
                'obs_spaces': obs_spaces,
                'recurrent': bool(args.recurrent_policy),
                'hidden_size': args.hidden_size,
                'critic_detach': args.critic_detach,
            },
        }
        options_policy = OptionsPolicy(**options_policy_kwargs)

    # Model for inferring options from trajectories
    trajectory_encoder_kwargs = {}
    trajectory_optim = None
    if args.model == 'hier' and args.hier_mode != 'infobot-supervised':
        traj_obs_spaces = obs_spaces.copy()
        traj_obs_spaces.pop('mission', None)
        if args.is_pomdp_env:
            traj_obs_spaces.pop('image', None)
            traj_obs_spaces.pop('direction', None)
            traj_obs_spaces['pos'] = pos_space

        # if args.ic_mode == 'diyan':
        #     assert args.traj_encoder_input == 'final_state'

        trajectory_encoder_kwargs = {
            'input_type': args.traj_encoder_input,
            'ic_mode': args.ic_mode,
            'observability': args.observability,
            'option_space': args.option_space,
            'omega_option_dims': args.omega_option_dims,
            'hidden_size': args.hidden_size,
            'base_model': args.base_model,
            'base_kwargs': {
                'obs_spaces': traj_obs_spaces,
                'recurrent': False,
                'hidden_size': args.hidden_size,
                'use_critic': False,
            },
        }
        trajectory_encoder = TrajectoryEncoder(**trajectory_encoder_kwargs)
        # if args.algo == 'acktr':
        #     from algo.kfac import KFACOptimizer
        #     trajectory_optim = KFACOptimizer(trajectory_encoder)
        # else:
        trajectory_optim = optim.RMSprop(
            trajectory_encoder.parameters(),
            # weight_decay=1e-3,
            alpha=args.alpha,
            lr=args.lr, eps=args.eps)
    else:
        trajectory_encoder = None
        # assert args.option_space == 'continuous'

    if args.hier_mode == 'vic':
        assert args.hr_goal_encoder_type == 'single'

    if args.flow_type != 'none':
        assert args.hr_goal_encoder_type != 'poe',\
            "POE with flows not supported"

    if args.model == 'cond':
        assert args.flow_type == 'none'

    # [NOTE] : OptionDecoder is depricated
    options_decoder_kwargs = {
        'observability': args.observability,
        'input_type': args.options_decoder_input,
        'encoder_type': args.hr_goal_encoder_type,
        'obs_spaces': obs_spaces,
        'attr_embed_size': args.attr_embed_size,
        'hidden_size': args.hidden_size,
        'option_space': args.option_space,
        'omega_option_dims': args.omega_option_dims,
        'base_model': args.base_model,
        'base_kwargs': {
            'obs_spaces': opt_dec_obs_spaces,
            'recurrent': False,
            'hidden_size': args.hidden_size,
            'use_critic': False,
        },
    }

    if args.model == 'hier' and args.hier_mode in ['vic', 'transfer', 'infobot-supervised'] or\
        args.model == 'cond' and args.hier_mode == 'bonus':
        options_decoder = DummyModule()
    else:
        # Options decoder
        if args.flow_type == 'planar':
            raise NotImplementedError
            # options_decoder = PlanarOptionsDecoder(
            #     num_flows=args.num_flows,
            #     options_decoder_kwargs=options_decoder_kwargs)

        elif args.flow_type == 'iaf':
            raise NotImplementedError
            # options_decoder = IAFOptionsDecoder(
            #     num_flows=args.num_flows,
            #     made_h_size=320,
            #     options_decoder_kwargs=options_decoder_kwargs)

        else:
            options_decoder = OptionsDecoder(**options_decoder_kwargs)

    if args.algo == 'ppo':
        raise NotImplementedError
        agent = algo.PPO(actor_critic=actor_critic,
                         ppo_version=args.ppo_version,
                         clip_param=args.clip_param,
                         ppo_epoch=args.ppo_epoch,
                         num_mini_batch=args.num_mini_batch,
                         value_loss_coef=args.value_loss_coef,
                         entropy_coef=args.entropy_coef,
                         lr=args.lr,
                         eps=args.eps,
                         max_grad_norm=args.max_grad_norm,
                         use_max_ent=args.use_max_ent,
                         max_ent_action_logprob_coeff=\
                         args.max_ent_action_logprob_coeff,
                         continuous_state_space=continuous_state_space,
                         model=args.model,
                         bonus_reward=bonus_reward)

    elif args.algo == 'a2c' or args.algo == 'acktr':
        use_entropy_reg = not args.use_max_ent
        if (args.hier_mode == 'bonus' or args.hier_mode == 'infobot-supervised') \
        and args.use_td:
            agent = algo.TD_A2C(
                actor_critic=actor_critic,
                options_policy=options_policy,
                # options_decoder=options_decoder,
                value_loss_coef=args.value_loss_coef,
                entropy_coef=args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                acktr=args.algo == 'acktr',
                normalize_advantage=args.normalize_advantage,
                use_entropy_reg=use_entropy_reg,
                max_ent_action_logprob_coeff=\
                args.max_ent_action_logprob_coeff,
                max_grad_norm=args.max_grad_norm,
                use_max_ent=args.use_max_ent,
                model=args.model,
                bonus_reward=bonus_reward,
                continuous_state_space=continuous_state_space,
                bonus_noise_scale=args.bonus_noise_scale,
                vis_env=vis_env
            )
        else:
            agent = algo.A2C_ACKTR(
                actor_critic=actor_critic,
                options_policy=options_policy,
                options_decoder=options_decoder,
                value_loss_coef=args.value_loss_coef,
                entropy_coef=args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                acktr=args.algo == 'acktr',
                normalize_advantage=args.normalize_advantage,
                use_entropy_reg=use_entropy_reg,
                max_ent_action_logprob_coeff=\
                args.max_ent_action_logprob_coeff,
                max_grad_norm=args.max_grad_norm,
                use_max_ent=args.use_max_ent,
                model=args.model,
                bonus_reward=bonus_reward,
                continuous_state_space=continuous_state_space,
                bonus_noise_scale=args.bonus_noise_scale,
                vis_env=vis_env
            )

    else:
        raise ValueError("Algo not supported {}".format(args.algo))


    if args.ac_start_from:
        # utils.load_from_ckpt(actor_critic, args.ac_start_from)
        utils.load_from_ckpt(
            args=args,
            # session_params=session_params,
            agent=agent,
            actor_critic=actor_critic,
            options_decoder=options_decoder,
            trajectory_encoder=trajectory_encoder,
            trajectory_optim=trajectory_optim,
            ckpt_path=args.ac_start_from)

        # Do not use loaded options decoder parameters
        # Initialize options_decoder from scratch
        # actor_critic.re_init_options_decoder(options_decoder_kwargs)

    ########################
    # SETUP ROLLOUTSTORAGE #
    ########################

    num_processes_eff = args.num_processes
    if (args.hier_mode == 'bonus' or args.hier_mode == 'infobot-supervised') \
    and args.use_td:
        rollouts = RolloutStorageTD(num_steps=args.num_steps,
                                  num_processes=num_processes_eff,
                                  obs_spaces=rollout_obs_spaces,
                                  recurrent_hidden_state_size=\
                                    actor_critic.recurrent_hidden_state_size,
                                  use_gae=args.use_gae,
                                  gamma=args.gamma,
                                  tau=args.tau,
                                  # omega_option_dims=args.omega_option_dims,
                                  z_latent_dims=args.z_latent_dims)
    else:
        rollouts = RolloutStorage(num_steps=args.num_steps,
                                  num_processes=num_processes_eff,
                                  obs_spaces=rollout_obs_spaces,
                                  recurrent_hidden_state_size=\
                                    actor_critic.recurrent_hidden_state_size,
                                  use_gae=args.use_gae,
                                  gamma=args.gamma,
                                  tau=args.tau,
                                  omega_option_dims=args.omega_option_dims,
                                  z_latent_dims=args.z_latent_dims)

    args_state = {
        'train_kwargs': train_kwargs,
        'options_decoder_kwargs': options_decoder_kwargs,
        'trajectory_encoder_kwargs': trajectory_encoder_kwargs,
    }
    if args.use_infobot:
        args_state['ib_policy_kwargs'] = ib_policy_kwargs
    else:
        args_state['policy_kwargs'] = policy_kwargs

    # if args.hier_mode == 'transfer':
    #     # Freeze actor_critic model
    #     print("[WARNING]: Freezing lower level actor-critic policy in "
    #           "tranfer mode")
    #     utils.freeze_params(actor_critic)

    def count_parameters(name, model):
        if model is not None:
            all_count = sum(p.numel() for p in model.parameters())
            trainable_count = sum(p.numel() \
                for p in model.parameters() if p.requires_grad)
            print("{}: {} / {}".format(name, trainable_count, all_count))

    print("="*80)
    print("Model / Trainable parameters / All parameters")
    count_parameters("actor_critic", actor_critic)
    count_parameters("trajectory_encoder", trajectory_encoder)
    count_parameters("options_policy", options_policy)
    count_parameters("options_decoder", options_decoder)
    print("="*80)

    if (args.hier_mode == 'bonus' or args.hier_mode == 'infobot-supervised') \
    and args.use_td:
        trainer = TrainerTD(
            args=args,
            train_envs=train_envs,
            val_envs=val_envs,
            vis_env=vis_env,
            actor_critic=actor_critic,
            options_policy=options_policy,
            options_decoder=options_decoder,
            # trajectory_encoder=trajectory_encoder,
            # trajectory_optim=trajectory_optim,
            z_encoder=z_encoder,
            b_args=b_args,
            args_state=args_state,
            agent=agent,
            rollouts=rollouts,
            device=device,
            num_processes_eff=num_processes_eff,
        )
    else:
        trainer = Trainer(
            args=args,
            train_envs=train_envs,
            val_envs=val_envs,
            vis_env=vis_env,
            actor_critic=actor_critic,
            options_policy=options_policy,
            options_decoder=options_decoder,
            trajectory_encoder=trajectory_encoder,
            trajectory_optim=trajectory_optim,
            z_encoder=z_encoder,
            b_args=b_args,
            args_state=args_state,
            agent=agent,
            rollouts=rollouts,
            device=device,
            num_processes_eff=num_processes_eff,
        )

    num_batches_per_epoch = 100
    # NUM_EPOCHS = 10000000
    NUM_EPOCHS = int(args.num_episodes) // \
        (num_batches_per_epoch * num_processes_eff)
    NUM_UPDATES = int(args.num_episodes) // num_processes_eff


    '''Use this for debugging NaN values in backward pass'''
    # with torch.autograd.detect_anomaly():
    #     print("[DEBUG] Using torch.autograd.detect_anomaly! Training will be slower.")
    #     trainer.train(start_iter=start_iter, num_epochs = NUM_EPOCHS)


    if args.hier_mode == 'infobot-supervised' and args.use_td:
        trainer.train_infobot_supervised(start_iter=start_iter,
            total_training_steps=int(args.total_training_steps))
    elif args.hier_mode == 'bonus' and args.use_td:
        trainer.train(start_iter=start_iter,
            total_training_steps=int(args.total_training_steps))
    else:
        trainer.train(start_iter=start_iter, num_epochs=NUM_EPOCHS)

    print("Training complete.")
