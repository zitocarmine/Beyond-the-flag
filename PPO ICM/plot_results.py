import pandas as pd
import matplotlib.pyplot as plt

FILE_CSV = "training_log_ppo.csv"
WINDOW = 20 

def plot_graphs():
    try:
        df = pd.read_csv(FILE_CSV)
    except FileNotFoundError:
        print(f"File {FILE_CSV} not found")
        return

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    axs[0].plot(df['Step'], df['Ext_Reward'], alpha=0.2, color='blue')
    axs[0].plot(df['Step'], df['Ext_Reward'].rolling(WINDOW).mean(), label='External Reward (Score)', color='blue', linewidth=2)
    axs[0].set_title("External Reward")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(df['Step'], df['Int_Reward'], alpha=0.2, color='orange')
    axs[1].plot(df['Step'], df['Int_Reward'].rolling(WINDOW).mean(), label='Intrinsic Reward (Curiosity)', color='orange', linewidth=2)
    axs[1].set_title("Curiosity Reward")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(df['Step'], df['Max_X_Pos'], alpha=0.3, color='green')
    axs[2].plot(df['Step'], df['Max_X_Pos'].rolling(WINDOW).max(), label='Max X Position', color='green', linewidth=2)
    axs[2].set_title("Map Exploration")
    axs[2].set_ylabel("Pixel X")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("grph_ppo_icm.png")
    print("Graph saved")
    plt.show()

if __name__ == "__main__":
    plot_graphs()