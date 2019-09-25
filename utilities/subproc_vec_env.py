import numpy as np
from multiprocessing import Process, Pipe
# from . import VecEnv, CloudpickleWrapper
# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    prev_obs = ob
                    prev_info = info
                    ob, info = env.reset()
                    info['prev_episode'] = {
                        'obs': prev_obs,
                        'info': prev_info,
                    }
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'reset_config_rng':
                env.reset_config_rng()
                remote.send(True)
            elif cmd == 'modify_attr':
                env.modify_attr(**data)
                # print("Worked modified attr: {attr} with value: {value}".format(**data))
                remote.send(None)
            elif cmd == 'get_attr':
                _attr = getattr(env, data)
                # print("Worker fetched attr {} with value {}".format(data, _attr))
                remote.send(_attr)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class SubprocVecEnvCustom(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def modify_attr(self, attr, values):
        self._assert_not_closed()
        for remote, value in zip(self.remotes, values):
            remote.send(('modify_attr', {'attr': attr, 'value': value}))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_config_rng(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset_config_rng', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_attr(self, attr):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_attr', attr))
        return np.stack([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def render(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
