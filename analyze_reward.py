import re
import matplotlib.pyplot as plt
import numpy as np

def extract_rewards(log_file_path):
    """
    Extract reward values from log file.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        list: List of extracted reward values
    """
    rewards = []
    
    # Regular expression to match "INFO -   Reward: X.XXX" and capture the number
    reward_pattern = re.compile(r'INFO -\s+Reward:\s+([-+]?\d*\.\d+|\d+)')
    
    # Read the file and extract rewards
    with open(log_file_path, 'r') as file:
        for line in file:
            match = reward_pattern.search(line)
            if match:
                reward_value = float(match.group(1))
                rewards.append(reward_value)
    
    return rewards

def plot_rewards(rewards, output_file='reward_plot.png', window_size=None):
    """
    Plot reward values and save the figure.
    
    Args:
        rewards (list): List of reward values
        output_file (str): Path to save the output plot
        window_size (int, optional): If provided, also plot moving average with this window
    """
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.plot(rewards, 'b-', alpha=0.3, label='Raw Rewards')
    
    # Add moving average if window_size is specified
    if window_size and len(rewards) > window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r-', 
                 linewidth=2, label=f'Moving Avg (window={window_size})')
    
    # Add horizontal lines every 28 units
    for i in range(0, len(rewards), 28):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Reward Values During Training')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add summary statistics as text
    if rewards:
        stats_text = (f"Mean: {np.mean(rewards):.4f}\n"
                     f"Median: {np.median(rewards):.4f}\n"
                     f"Min: {np.min(rewards):.4f}\n"
                     f"Max: {np.max(rewards):.4f}\n"
                     f"Total Episodes: {len(rewards)}")
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    
    # Also display it if running in a notebook or interactive environment
    plt.show()

def main():
    # Path to your log file
    log_file_path = "/home/gridsan/mhadjiivanov/meng/G-Retriever/webqsp_new_reinforce_pretrain_load.log"
    
    # Extract rewards
    rewards = extract_rewards(log_file_path)
    
    if not rewards:
        print("No reward values found in the log file!")
        return
    
    print(f"Extracted {len(rewards)} reward values")
    
    # Plot rewards with a moving average window of 10
    plot_rewards(rewards, window_size=10)

if __name__ == "__main__":
    main()