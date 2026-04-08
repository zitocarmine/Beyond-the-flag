import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import multiprocessing as mp
import argparse
import os
import csv
from env_wrapper import create_mario_env
from model_a2c import MarioA2C

NUM_ENVS = 32          
NUM_STEPS = 20          
TOTAL_STEPS = 50000000 
LR = 1.0e-4            
GAMMA = 0.99

ICM_SCALE = 25.0      
ENT_START = 0.25     
ENT_END = 0.01
ENT_DECAY = 6000000  

SAVE_FILE = "mario_a2c_mastery.pth"
LOG_FILE = "training_log_a2c.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def worker_process(remote, parent_remote):
    parent_remote.close()
    env = create_mario_env(render=False)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, reward, done, trunc, info = env.step(data)
                x_pos = info.get('x_pos', 0)
                if done or trunc: obs, _ = env.reset()
                remote.send((obs, reward, done or trunc, x_pos))
            elif cmd == 'reset':
                obs, _ = env.reset()
                remote.send(obs)
            elif cmd == 'close':
                break
    except KeyboardInterrupt:
        pass
    finally:
        remote.close()

class ParallelEnv:
    def __init__(self, n):
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n)])
        self.ps = [mp.Process(target=worker_process, args=(w, r)) for w, r in zip(self.work_remotes, self.remotes)]
        for p in self.ps: p.daemon = True; p.start()
        for w in self.work_remotes: w.close()
    
    def step(self, actions):
        for r, a in zip(self.remotes, actions): r.send(('step', a))
        results = [r.recv() for r in self.remotes]
        obs, rews, dones, x_poss = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(x_poss)
    
    def reset(self):
        for r in self.remotes: r.send(('reset', None))
        return np.stack([r.recv() for r in self.remotes])
    
    def close(self):
        for r in self.remotes: r.send(('close', None))
        for p in self.ps: p.join()

def train():
    print(f"Start Training A2C+ICM on {DEVICE} with {NUM_ENVS} workers...")
    
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Ext_Reward", "Int_Reward", "Max_X_Pos", "Entropy"])

    envs = ParallelEnv(NUM_ENVS)
    temp_env = create_mario_env()
    n_actions = temp_env.action_space.n
    agent = MarioA2C(n_actions).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LR)
    
    step = 0
    if os.path.exists(SAVE_FILE):
        print(f"Loading {SAVE_FILE}...")
        ckpt = torch.load(SAVE_FILE, map_location=DEVICE)
        agent.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        step = ckpt.get('step', 0)

    obs = envs.reset()
    hx = torch.zeros(NUM_ENVS, 512).to(DEVICE)
    cx = torch.zeros(NUM_ENVS, 512).to(DEVICE)
    
    try:
        while step < TOTAL_STEPS:
            current_ent = ENT_START - (ENT_START - ENT_END) * min(1.0, step / ENT_DECAY)

            b_obs, b_next_obs, b_act, b_rew, b_val, b_don = [], [], [], [], [], []
            b_hx, b_cx = [], [] 
            
            temp_ext_rews = []
            temp_int_rews = []
            temp_max_x = []

            for _ in range(NUM_STEPS):
                step += NUM_ENVS
                t_obs = torch.tensor(obs).float().to(DEVICE)
                
                b_hx.append(hx.detach().cpu().numpy())
                b_cx.append(cx.detach().cpu().numpy())

                with torch.no_grad():
                    logits, val, next_hx, next_cx = agent(t_obs, hx, cx)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()

                next_obs, ext_rew, dones, x_poss = envs.step(action.cpu().numpy())
                
                with torch.no_grad():
                    t_next = torch.tensor(next_obs).float().to(DEVICE)
                    _, fwd_err = agent.get_icm_loss(t_obs, t_next, action, n_actions)
                    int_rew = fwd_err.cpu().numpy() * ICM_SCALE
                
                total_rew = ext_rew + int_rew

                temp_ext_rews.append(np.mean(ext_rew))
                temp_int_rews.append(np.mean(int_rew))
                temp_max_x.append(np.max(x_poss))

                b_obs.append(obs)
                b_next_obs.append(next_obs)
                b_act.append(action.cpu().numpy())
                b_rew.append(total_rew)
                b_val.append(val.view(-1).cpu().numpy())
                b_don.append(dones)
                
                obs = next_obs
                mask = torch.tensor(1.0 - dones).float().to(DEVICE).view(-1, 1)
                hx = next_hx * mask; cx = next_cx * mask

            with torch.no_grad():
                next_val = agent(torch.tensor(obs).float().to(DEVICE), hx, cx)[1].view(-1).cpu().numpy()
                adv = np.zeros_like(b_rew)
                lastgaelam = 0
                for t in reversed(range(NUM_STEPS)):
                    nextnonterminal = 1.0 - b_don[t]
                    nextvalues = next_val if t == NUM_STEPS - 1 else b_val[t + 1]
                    delta = b_rew[t] + GAMMA * nextvalues * nextnonterminal - b_val[t]
                    adv[t] = lastgaelam = delta + GAMMA * 0.95 * nextnonterminal * lastgaelam
                returns = adv + b_val

            f_obs = torch.tensor(np.array(b_obs).reshape(-1, 4, 84, 84)).float().to(DEVICE)
            f_next_obs = torch.tensor(np.array(b_next_obs).reshape(-1, 4, 84, 84)).float().to(DEVICE)
            f_act = torch.tensor(np.array(b_act).flatten()).long().to(DEVICE)
            f_ret = torch.tensor(returns.flatten()).float().to(DEVICE)
            f_adv = torch.tensor(adv.flatten()).float().to(DEVICE)
            f_hx = torch.tensor(np.array(b_hx).reshape(-1, 512)).float().to(DEVICE)
            f_cx = torch.tensor(np.array(b_cx).reshape(-1, 512)).float().to(DEVICE)

            logits, val, _, _ = agent(f_obs, f_hx, f_cx)
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(f_act)
            entropy = dist.entropy().mean()
            
            policy_loss = -(log_prob * f_adv).mean()
            
            value_loss = 0.5 * ((val.view(-1) - f_ret) ** 2).mean()
            
            inv_l, fwd_l = agent.get_icm_loss(f_obs, f_next_obs, f_act, n_actions)
            
            loss = policy_loss + 0.5 * value_loss - current_ent * entropy + fwd_l.mean() + inv_l.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

            mean_ext = np.mean(temp_ext_rews)
            mean_int = np.mean(temp_int_rews)
            max_x = np.max(temp_max_x)

            if step % 1000 == 0: 
                print(f"Step {step} | Ext: {mean_ext:.3f} | Int: {mean_int:.3f} | Max X: {max_x:.1f} | Ent: {current_ent:.3f}")
            
            with open(LOG_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, mean_ext, mean_int, max_x, current_ent])

            if step % 100000 < NUM_STEPS * NUM_ENVS:
                torch.save({'model': agent.state_dict(), 'optim': optimizer.state_dict(), 'step': step}, SAVE_FILE)
                print("Autosave.")

    except KeyboardInterrupt:
        print("Saving")
        torch.save({'model': agent.state_dict(), 'optim': optimizer.state_dict(), 'step': step}, SAVE_FILE)
    finally:
        envs.close()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', required=True)
    args = parser.parse_args()

    if args.train: train()