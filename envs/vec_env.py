import copy
import omnigibson as og
from omnigibson.envs import VectorEnvironment
from envs.base_env import BaseEnv
from tqdm import trange


class VecEnv(VectorEnvironment):
    """
    Define vectorized environment class, inheriting from omnigibson's VectorEnvironment class
    """

    def __init__(self, num_envs, config):
        self.num_envs = num_envs
        if og.sim is not None:
            og.sim.stop()

        # First we create the environments. We can't let DummyVecEnv do this for us because of the play call
        # needing to happen before spaces are available for it to read things from.
        self.envs = [
            BaseEnv(config=copy.deepcopy(config), in_vec_env=True)
            for _ in trange(num_envs, desc="Loading environments")
        ]

        # Play, and finish loading all the envs
        og.sim.play()
        for env in self.envs:
            env.post_play_load()
