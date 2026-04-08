import torch
import numpy as np
import time
import os
from env_wrapper import create_mario_env
from model_a2c import MarioA2C

MODEL_PATH = "mario_a2c_pure.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FPS = 30 

def play():
    print(f"loading Mario Pure A2C on {DEVICE}")
    
    env = create_mario_env(render=True)
    n_actions = env.action_space.n
    
    model = MarioA2C(n_actions).to(DEVICE)
    model.eval() 
    
    if os.path.exists(MODEL_PATH):
        print(f"CLoading model from {MODEL_PATH}...")
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model'])
            print(f"Model properly loaded (Step: {checkpoint.get('step', 'unknown')})")
        except Exception as e:
            print(f"ERROR: {e}")
            return
    else:
        print(f"ERRORE: Model not found {MODEL_PATH}.")
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
        print("\nClosing")
        env.close()

if __name__ == "__main__":
    play()