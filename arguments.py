import os
import argparse
from time import gmtime, strftime

import torch

def get_args(mode: str = 'train'):
    assert mode in ['train', 'eval']
    parser = argparse.ArgumentParser(description='RL')

    # Input file specifications
    parser.add_argument('--env_root',

            help='Directory containing all environment files.')

    parser.add_argument('--abs_env_root',

            help='Directory containing environment files for abstraction'
            'split')

    parser.add_argument('--splits_dir',default='../splits/simple_grid_world/v1.0',
            help="Diretory with train, val and test splits for attrs and envs."
            "val splits should be under directory 'val' and test splits"
            "should be under directory 'test' ")
    parser.add_argument('--train_split',default='train.json',
            help='Split of the environment used for training.')
    parser.add_argument('--pov_imgs_path',

            help="HDF5 file with POV images of all environments.")

    if mode == 'eval':
        parser.add_argument('--load_path',type=str,default="",
                            help='Checkpoint path for loading.' )


    # Output file specifications
    parser.add_argument('--viz_dir',default='data/viz',
            help='Visualization files are saved here.')
    parser.add_argument('--log-dir', default='logs/',
                        help='directory to save agent logs (default: logs/)')
    parser.add_argument('--save-dir', default='checkpoints/',
                        help='directory to save model checkpoints.')
    parser.add_argument('--save-sub-dir', default='',
                        help='Save sub-directory. If empty string, then'
                             'sub-directory name is auto generated.')
    parser.add_argument('--ac_start_from', default="",
                        help='Ckpt to start actor_critic.')
    parser.add_argument('--
                        help='Ckpt to load enc model for KL bonus.')
    # parser.add_argument('--encoder_checkpoint', default=\



    # Optimization parameters
    parser.add_argument('--algo', default='ppo', choices=['ppo', 'a2c', 'acktr'],
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--normalize_advantage', default=0, type=int,
                        help='whether to batch normalize advantage')
    parser.add_argument('--ppo-version', default='default',
                        choices=['default', 'ppo-with-options'],
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.97,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', type=int, default=1,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--clip_param', type=float, default=0.1,
                        help='ppo clip parameter (default: 0.1)')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--critic_detach', type=int, default=1,
                        help='whether to detach critic input')

    # Train-time parameters
    parser.add_argument('--num_agents',type=int,default=5,
            help='Number of agents used for training.' )
    parser.add_argument('--num_processes', type=int, default=12,
            help='how many training CPU processes to use (default: 4)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-episodes', type=float, default=10e7,
                        help='number of episodes to train (default: 10e7)')
    parser.add_argument('--total_training_steps', type=float, default=10e10,
                        help='number of time steps to train TD (default: 10e10)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=8,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--dataloader_workers', default=16, type=int)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')


    # Environment/dataset choices
    parser.add_argument('--env-name', default='twogoals',
            choices=['twogoals', 'randomtwo', 'onecorner', 'corridor',
                    'multiroom', 'multiroom-mdp', 'two-room-corridor',
                    'four-rooms', 'maze', 'crossing', 'crossing-mdp',
                    'pillar-grid', 'empty-pomdp', 'two-room-corridor-pomdp',
                    'four-rooms-pomdp', 'maze-pomdp', 'mountain-car',
                    'acrobat'],
            help='environment to train on (default: twogoals)')
    parser.add_argument('--env_grid_size', type=int, default=5)
    parser.add_argument('--obs_win_size', type=int, default=1)
    parser.add_argument('--agent_view_size', type=int, default=7,
            help='partially observable view size of agent in pomdp')
    parser.add_argument('--end_on_goal', type=int, default=0,
            help="Whether to end the episode when agent hits goal.")
    parser.add_argument('--num-steps', type=int, default=50,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--num_option_steps', type=int, default=10,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--replan_strategy', type=str, default='constant',
                        choices=['constant', 'decision_state'],
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--max_targets_per_env', type=int, default=0,
           help="Number of possible target objects per env, <=0"
           "to select all objects as possible targets.")
    parser.add_argument('--max_train_envs',type=int,default=0,
            help='Maximum number of envs used for training. <=0 to select all.')
    parser.add_argument('--max_val_envs',type=int,default=-1,
            help='Maximum number of envs used for validation, <=0 to select all.')
    parser.add_argument('--max_reachability_threshold', type=int, default=4,
            help='Max distance from target that is considered as success for navigation')

    parser.add_argument('--reset_prob', type=float, default=1.0,
            help='Probability of reset after option termination')
    parser.add_argument('--perturb_prob', type=float, default=0.0,
            help='Probability of agent position perturbation after every step')
    parser.add_argument('--reset_adaptive', type=int, default=0,
            help='Reset an agent based on IC performance')
    parser.add_argument('--term_prob', type=float, default=0.0,
            help='Probability of episode termination at each step')
    parser.add_argument('--terminal_reward', type=int, default=0,
            help='Terminal reward used in case of sparse rewards.')
    parser.add_argument('--randomize_goal_pos', type=int, default=0,
            help='Randomly spawn goal in env')

    parser.add_argument('--corridor_len', type=int, default=1,
            help="Length of corridor in the two-room-corridor environment.")
    parser.add_argument('--multiroom_num_rooms', type=int, default=2,
            help="Number of rooms in multiroom env.")
    parser.add_argument('--multiroom_doors_open', type=int, default=0,
            help="Whether to keep multiroom doors open or closed on reset.")
    parser.add_argument('--maze_complexity', type=float, default=0.9,
            help="Complexity argument for maze env.")
    parser.add_argument('--maze_density', type=float, default=0.9,
            help="Density argument for maze env.")
    parser.add_argument('--config_seed', type=int, default=13,
            help="Seed for the environment configuration")
    parser.add_argument('--static_env_grid', type=int, default=1,
            help="Whether env is randomized on every reset or static.")

    parser.add_argument('--train_reachability_threshold', type=int, default=4,
            help='Padding for the goal object bounding box. The padded goal '
            'bounded box is considered as goal boundry.')

    parser.add_argument('--test_recall', type=int, default=0,
            help='Set true to test recall on validation split.')

    parser.add_argument('--num_eval_episodes', type=int, default=100,
            help='Number of episodes to run evaluation of success on.')

    # Goal Specific Parameters
    parser.add_argument('--goal_embed_size', type=int, default=32,
            help='Size of the embedding table for each attribute value for the'
            'goal')
    parser.add_argument('--goal_hidden_size',type=int, default=64,
            help='Hidden layer size for the FC layer after concatenation of'
            'individual attribute value embedding.')
    parser.add_argument('--goal_output_size', type=int,default=32,
            help='Size of final goal encoding used as a representation for'
            'the goal everywhere in the training.')

    # State Encoder Parameters
    parser.add_argument('--state_encoder_hidden_size',type=int,default=128,
            help='Hidden layer size used in MLP after CNN processing of the state'
            'map of the environment')


    # Architecture parameters
    parser.add_argument('--observations', default=3,
            choices=[1,3],
            help="1 for standard observation space, 3 for triplet observations ")
    parser.add_argument('--recurrent_policy', default=0, type=int,
                        help='use a recurrent policy')
    parser.add_argument('--recurrent_encoder', default=0, type=int,
                        help='use a recurrent z encoder')
    parser.add_argument('--allow_early_stop', type=int, default=1,
           help="Allow the agent to end episode with 'stay' option")
    parser.add_argument('--use_pretrained_encoder', default=1, type=int,
            help="Whether to use a pre-trained CNN encoder")
    parser.add_argument('--use_td', default=0, type=int,
            help="TD Updates instead of monte carlo")

    parser.add_argument('--grid_obs_type', default='xy',
            choices=['xy', 'grid'],
            help="Whether to use grid in state observation")


    parser.add_argument('--observability',default='full',
            choices=('full','partial','partial-triplet'),
            help="'full' for MDP case, 'partial' for POMDP case with single"+
            "observation space and 'partial-triplet' for triplet observation space")

    parser.add_argument('--model', default='cond',
            choices=('cond', 'hier'),
            help="'cond' model only taken in goal specification to"
            "parameterize policy. 'hier' model also uses the options for"
            "parameterization")

    parser.add_argument('--hier_mode', default='default',
            choices=['default', 'vic', 'transfer', 'bonus', 'infobot-supervised'])

    parser.add_argument('--bonus_type', default='count',
            choices=['count', 'kl', 'kl-pi'])

    parser.add_argument('--bonus_noise_scale', default=0.0, type=float)

    parser.add_argument('--bonus_normalization', default='unnormalized',
            choices=['unnormalized'])

    parser.add_argument('--ic_mode', default='default',
            choices=['vic', 'diyan', 'valor'])

    parser.add_argument('--use_infobot', default=0, type=int,
            help="Whether to use an infobot policy (i.e. with a z_t) or a regular policy")

    parser.add_argument('--options_decoder_input',
            default='goal_and_initial_state',
            choices=['goal_only', 'goal_and_initial_state'],
            help="Options decoder inputs - can either condition on"
                "just the goal specification or additionally the"
                "initial state observation")

    parser.add_argument('--closed_loop', type=int, default=1,
           help="Use trajectory encoder to predict omega")

    parser.add_argument('--traj_encoder_input', default='final_and_initial_state',
           choices = ['final_state', 'final_and_initial_state'],
           help="Use trajectory encoder input type")
    parser.add_argument('--traj_enc_loss_coeff', type=float, default=1.0)

    # Max-Ent
    parser.add_argument('--max_ent_action_logprob_coeff', type=float, default=0.01,
            help="Weight used for negative action logprob when added to the reward"+
            "to get max_end policy gradients in PPO.")

    parser.add_argument('--use_max_ent', type=int, default=0,
            help="Use max entropy objective for control." )

    # [NOTE] : Action space only defined for 'args.observability = full' for now.

    # For 'args.observability = partial', two additional action spaces are
    # needed. We can add that later.
    parser.add_argument('--action_space_id', default=0, type=int,
            help="0 - Action space for 'full' observability")

    # -base_model can be inferred from observability
    parser.add_argument('--base_model', default='mlp',
           choices=['cnn-mlp', 'mlp'],
           help="'cnn-mlp' - First use CNN layers to process the obeservation and then pass"
                "the features through an MLP"
                "'mlp' - flatten the observation and pass through an MLP")

    parser.add_argument('--target_embed_type', default='k-hot',
            choices=['k-hot'],
            help="Target embedding type for cnn")

    parser.add_argument('--attr_embed_size', type=int, default=32)
    parser.add_argument('--hr_goal_encoder_type', default='single',
            choices=['single', 'poe'])
    parser.add_argument('--option_space', default='continuous',
            choices=['continuous', 'discrete'])
    parser.add_argument('--omega_option_dims', type=int, default=32)
    parser.add_argument('--z_latent_dims', type=int, default=32)
    parser.add_argument('--z_stochastic', type=int, default=1)
    parser.add_argument('--z_std_clip_max', type=float, default=2,
            help="Max value to clip z standard deviation to when sampling.")
    parser.add_argument('--flow_type', default='none',
            choices=['none', 'planar', 'iaf'])
    parser.add_argument('--num_flows', type=int, default=5)

    #[NOTE] : Important hyperparameter to tune
    parser.add_argument('--kl_optim_mode', default='analytic',
            choices=['analytic', 'mc_sampling'])
    parser.add_argument('--ib_kl_mode', default='analytic',
            choices=['analytic', 'mc_sampling'])
    parser.add_argument('--hr_model_kl_coeff', type=float, default=1.0)
    parser.add_argument('--use_omega_dim_curriculum', type=int, default=0)
    parser.add_argument('--omega_curr_win_size', type=int, default=200)
    parser.add_argument('--omega_traj_ll_theta', type=float, default=0.86)
    parser.add_argument('--reweight_by_omega_ll', type=int, default=0)
    parser.add_argument('--infobot_beta', type=float, default=0.01)
    parser.add_argument('--infobot_beta_min', type=float, default=1e-6)
    parser.add_argument('--infobot_beta_curriculum', type=str, default='0-0')
    parser.add_argument('--ib_adaptive', type=int, default=0)
    parser.add_argument('--q_start_epochs', type=int, default=10)
    parser.add_argument('--kl_anneal_start_epochs', type=int, default=10)
    parser.add_argument('--kl_anneal_growth_epochs', type=int, default=10)
    parser.add_argument('--spawn_curriculum', default='random',
            choices=['random', 'center', 'fixed'])

    parser.add_argument('--bonus_beta', type=float, default=1.0,
            help="Coefficient for reward bonus.")
    parser.add_argument('--bonus_heuristic_beta', type=float, default=0.0,
            help="Coefficient for reward bonus.")
    parser.add_argument('--reward_scale', default=1.0, type=float)
    parser.add_argument('--reward_type', default='neg_l2',
            choices=['neg_l2', 'dense_l2', 'neg_dense_l2',
                        'target_reached', 'exp_l2', 'dense_binary_v1',
                        'dense_binary_v2', 'neg_l1', 'dense_l1',
                        'pot_diff_reshaped', 'dense_spiky_l2',
                        'sparse_spiky_l2', 'dense_l1_xpe'])

    parser.add_argument('--potential_type', default='l1',
            choices=['l1', 'l2'])

    parser.add_argument('--spike_value', default=5, type=int)

    parser.add_argument('--hidden_size', type=int, default=128)


    # Logging/bookkeeping parameters
    parser.add_argument('--identifier', default='nav-main',
            help='model type identifier, currently unused')
    parser.add_argument('--visdom_env_name', default='main')
    parser.add_argument('--vis', type=int, default=1,
            help='enable visdom visualization')
    parser.add_argument('--server', type=str, default='http://localhost',
            help='visdom server address')
    parser.add_argument('--port', type=int, default=8893,
            help='port to run the server on (default: 8893)')
    parser.add_argument('--log-interval', type=int, default=10,
            help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--heatmap-interval', type=int, default=10,
            help='heatmap plotting interval (in epochs)')
    parser.add_argument('--save-interval', type=int, default=10,
            help='save interval, one save per n epochs (default: 10)')
    parser.add_argument('--val_interval',type=int,default=10,
            help="Validation to be done after these many epochs.")
    parser.add_argument('--skip_eval', type=int, default=0,
            help="Skip eval, use for debugging.")
    parser.add_argument('--vid_interval', type=int, default=100,
            help="Video vis interval")
    parser.add_argument('--vid_envs',type=int,default=5,
           help='Number of environments to use for videos.')
    parser.add_argument('--vid_agents',type=int, default=2,
           help='Number of agents frome ach env to use for videos.')
    parser.add_argument('--vis_heatmap', type=int, default=1,
           help='Visualize agent s_initial and s_final heatmap')

    # parser.add_argument('--logging_decay_coeff', type=float, default=0.98,
    #         help="Decay for moving averages in visdom line plots")

    # parser.add_argument('-obs-embed-size', type=int, default=1)
    #parser.add_argument('-mask_actions_on_done', type=int, default=0,
    #        help="NOTE: THIS OPTION IS NOW REDUNDANT!")

    # parser.add_argument('--eval-interval', type=int, default=None,
    #                     help='eval interval, one eval per n updates (default: None)')
    # parser.add_argument('--vis-interval', type=int, default=10,
    #                     help='vis interval, one log per n updates (default: 100)')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    def parse_curriculum_str(_str):
        _split = _str.split("-")
        if len(_split) == 3:
            assert _split[0] == 'auto'
            AUTO_KL = True
            return AUTO_KL, int(_split[1]), int(_split[2])
        else:
            AUTO_KL = False
            return [AUTO_KL] + [int(item) for item in _split]

    _auto_kl, _start, _growth = parse_curriculum_str(args.infobot_beta_curriculum)
    args.infobot_auto_kl = _auto_kl
    args.infobot_kl_start = _start
    args.infobot_kl_growth = _growth

    import random
    if args.save_sub_dir:
        args.save_dir = os.path.join(args.save_dir, args.save_sub_dir)
    else:
        timeStamp = strftime('%d-%b-%Y-%H-%M-%S', gmtime())
        args.save_dir = os.path.join(args.save_dir, timeStamp)
        args.save_dir += '_{:0>6d}'.format(random.randint(0, 10e6))

    return args
