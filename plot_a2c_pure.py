import pandas as pd
import matplotlib.pyplot as plt
import os

FILE_CSV = "training_log_a2c_pure.csv"
WINDOW = 50  
OUTPUT_IMG = "graph_a2c_pure.png"

def plot_graphs():
    if not os.path.exists(FILE_CSV):
        print(f"ERROR: File {FILE_CSV} not found")
        return

    try:
        df = pd.read_csv(FILE_CSV)
    except Exception as e:
        print(f"Error in CSV: {e}")
        return


    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    axs[0].plot(df['Step'], df['Ext_Reward'], alpha=0.15, color='blue')
    axs[0].plot(df['Step'], df['Ext_Reward'].rolling(WINDOW).mean(), label='External Reward (Score)', color='blue', linewidth=2)
    axs[0].set_title("External Reward")
    axs[0].set_ylabel("Score / Reward")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(df['Step'], df['Max_X_Pos'], alpha=0.2, color='green')
    axs[1].plot(df['Step'], df['Max_X_Pos'].rolling(WINDOW).max(), label='Max X Position', color='green', linewidth=2)
    axs[1].set_title("Map Exploration")
    axs[1].set_xlabel("Training Steps")
    axs[1].set_ylabel("Pixel X Coordinate")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"Graph saved in '{OUTPUT_IMG}'")
    plt.show()

if __name__ == "__main__":
    plot_graphs()