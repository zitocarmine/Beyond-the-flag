import torch
import numpy as np
import time
from env_wrapper import create_mario_env
from model_a2c import MarioA2C
import os

MODEL_PATH = "mario_a2c_mastery.pth"  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FPS = 30 

def play():
    print(f"Starting Mario A2C+ICM on {DEVICE}...")
    
    env = create_mario_env(render=True)
    model = MarioA2C(env.action_space.n).to(DEVICE)
    model.eval() 
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
        print(f"Model loaded (Step: {checkpoint.get('step', '?')})")
    else:
        print(f"ERROR: model not found {MODEL_PATH}")
        return

    try:
        while True:
            obs, _ = env.reset()
            hx = torch.zeros(1, 512).to(DEVICE)
            cx = torch.zeros(1, 512).to(DEVICE)
            done = False
            total_reward = 0
            
            while not done:
                state = torch.tensor(np.array(obs)).float().to(DEVICE).unsqueeze(0)
                
                with torch.no_grad():
                    logits, val, hx, cx = model(state, hx, cx)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample() 

                obs, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated
                total_reward += reward
                
                time.sleep(1.0 / FPS)
                env.render()

            print(f"GAME OVER: Total reward: {total_reward:.2f} | Max distance reached: {info.get('x_pos', 0)}")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nClosing...")
        env.close()

if __name__ == "__main__":
    play()