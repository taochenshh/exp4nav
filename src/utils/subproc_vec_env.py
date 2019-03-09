from multiprocessing import Process, Pipe

import numpy as np

from . import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            if isinstance(data, dict):
                ob = env.reset(**data)
            else:
                ob = env.reset()
            remote.send(ob)
        elif cmd == 'render':
            env.render()
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        print('Number of envs:', nenvs)
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
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        if isinstance(obs[0], tuple):
            pobs = zip(*obs)
            xobs = [np.stack(x) for x in pobs]
            return xobs, np.stack(rews), np.stack(dones), infos
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, conds=None):
        if conds is None:
            conds = [None] * len(self.remotes)
        for remote, cond in zip(self.remotes, conds):
            remote.send(('reset', cond))
        obs = [remote.recv() for remote in self.remotes]
        if isinstance(obs[0], tuple):
            pobs = zip(*obs)
            xobs = [np.stack(x) for x in pobs]
            return xobs
        return np.stack(obs)

    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
