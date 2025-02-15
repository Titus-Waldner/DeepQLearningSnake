"""
Plot stats from stats.txt

This script reads stats.txt (which should contain lines like "Episode,MovingAvgScore")
and plots the Moving Average Score vs. Episode.
"""

import matplotlib.pyplot as plt

STATS_FILE = "stats.txt"

episodes = []
moving_avg_scores = []

with open(STATS_FILE, "r") as f:
    header = f.readline()  # Skip header.
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 2:
            episode, score = parts
            episodes.append(int(episode))
            moving_avg_scores.append(float(score))

plt.figure(figsize=(10, 5))
plt.plot(episodes, moving_avg_scores, label="Moving Avg Score")
plt.xlabel("Episode")
plt.ylabel("Moving Average Score")
plt.title("Training Performance")
plt.legend()
plt.grid(True)
plt.show()
