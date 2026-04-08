import gym
import numpy as np
import cv2
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import warnings

warnings.filterwarnings("ignore")

if not hasattr(np, "bool"): np.bool = bool
if not hasattr(np, "float"): np.float = float

class ExplorerMarioReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._current_score = 0
        self._current_coins = 0
        self._last_x = 0
        self._last_status = 'small'
        
        self._time_penalty = -0.01 
        self._flag_reward = 500.0  
        self._death_penalty = -50.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_score = info.get('score', 0)
        self._current_coins = info.get('coins', 0)
        self._last_x = info.get('x_pos', 0)
        self._last_status = info.get('status', 'small')
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        score = info.get('score', 0)
        coins = info.get('coins', 0)
        x_pos = info.get('x_pos', 0)
        status = info.get('status', 'small')
        flag_get = info.get('flag_get', False)
        
        custom_reward = 0.0

        
        custom_reward += self._time_penalty 
        
        if score > self._current_score:
              custom_reward += (score - self._current_score) * 0.1
        
        if coins > self._current_coins:
              custom_reward += (coins - self._current_coins) * 2.0 
        
        if status != self._last_status:
              if status in ['tall', 'fireball'] and self._last_status == 'small':
                  custom_reward += 150.0 
              self._last_status = status

        v_x = x_pos - self._last_x
        if v_x > 0:
              custom_reward += v_x * 0.1
        self._last_x = x_pos

        if flag_get:
            custom_reward += self._flag_reward
            print("FLAG CAPTURED")
            terminated = True
        
        if terminated or info.get('life', 0) < 2: 
            custom_reward += self._death_penalty

        self._current_score = score
        self._current_coins = coins
        
        return obs, custom_reward, terminated, truncated, info

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
    def step(self, action):
        total_rew = 0.0
        done = False
        for _ in range(self._skip):
            obs, rew, term, trunc, info = self.env.step(action)
            total_rew += rew
            if term or trunc:
                done = True
                break
        return obs, total_rew, done, trunc, info

class GrayScaleResize(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)[None, :, :]

class GymCompat(gym.Wrapper):
    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        return (res[0], res[1]) if isinstance(res, tuple) else (res, {})
    def step(self, action):
        res = self.env.step(action)
        if len(res) == 4: return res[0], res[1], res[2], False, res[3]
        return res

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = np.zeros((k, 84, 84), dtype=np.uint8)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(k, 84, 84), dtype=np.uint8)
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames[:] = obs
        return self.frames.copy(), info
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = obs
        return self.frames.copy(), reward, term, trunc, info

def create_mario_env(render=False):
    mode = 'human' if render else 'rgb_array'
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode=mode)
    env = JoypadSpace(env, COMPLEX_MOVEMENT) 
    env = GymCompat(env)
    env = ExplorerMarioReward(env)
    env = SkipFrame(env, skip=4)
    env = GrayScaleResize(env)
    env = FrameStack(env, k=4)
    return env