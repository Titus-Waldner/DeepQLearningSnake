"""
Snake RL with DQN using PyTorch and PyGame (with CUDA support)

Features in this version:
 - Saves model weights every 100 episodes.
 - Toggle to load pre-trained weights at the start of training.
 - Adds a per-move time penalty and Euclidean distance–based reward shaping.
   In this version, we reward the agent only once per tile improvement:
     • When the snake first moves into a tile (as defined by the integer part of the Euclidean distance)
       closer to the food than before, it gets a bonus.
     • Subsequent moves that do not reduce the tile value get a penalty.
 - If the snake does not eat the fruit within 200 moves, the episode terminates with a large negative reward.
 - Every 100 episodes, if the moving average score is very low and epsilon is very low,
   epsilon is reset to 0.5 for additional exploration.
 - Logs episode number and moving average score to a stats file. If the default "stats.txt" cannot be written,
   it will try "stats_1.txt", "stats_2.txt", etc., and then use that file throughout training.

Before running:
 - Install dependencies: pip install torch pygame numpy
"""

import random
import numpy as np
import pygame
import sys
import time
from collections import deque
import os

import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------
# Parameters and Hyperparameters
# --------------------------
BLOCK_SIZE = 20        # Size of each block in pixels
GAME_WIDTH = 20        # Number of blocks in horizontal direction
GAME_HEIGHT = 20       # Number of blocks in vertical direction
FPS = 60               # Frames per second for visualization

MEMORY_SIZE = 2000     # Capacity of replay memory
BATCH_SIZE = 64        # Mini-batch size for training
GAMMA = 0.95           # Discount factor for future rewards
LEARNING_RATE = 0.0005 # Learning rate for optimizer
EPSILON_START = 0.3    # Starting value for epsilon
EPSILON_MIN = 0.01     # Minimum epsilon
EPSILON_DECAY = 0.999995  # Decay factor for epsilon per step

NUM_EPISODES = 50000   # Number of episodes for training
MOVE_LIMIT = 200       # Hard move limit per episode

# For temporal stacking: number of consecutive states to stack.
STACK_SIZE = 3

TRAIN_MODE = True      # True for training mode, False for play mode.
LOAD_PRETRAINED = True  # Set to True to load pre-trained weights at start.

MODEL_FILE = "snake_dqn_model.pth"
MODEL_FILE_TRAIN = "snake_dqn_model.pth"
BASE_STATS_FILE = "stats.txt"

# --------------------------
# Helper for Stats File
# --------------------------
def get_stats_filename(base_name):
    """
    Try to open base_name for writing; if fails, try base_name with an appended number.
    Returns the filename that can be successfully opened for writing.
    """
    i = 0
    while True:
        if i == 0:
            filename = base_name
        else:
            filename = f"stats_{i}.txt"
        try:
            with open(filename, "w") as f:
                f.write("Episode,MovingAvgScore\n")
            print(f"Stats will be logged to {filename}")
            return filename
        except PermissionError:
            i += 1

STATS_FILE = get_stats_filename(BASE_STATS_FILE)

# --------------------------
# Check for CUDA
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------
# Environment: Snake Game
# --------------------------
class SnakeGame:
    def __init__(self, width=GAME_WIDTH, height=GAME_HEIGHT):
        self.width = width
        self.height = height
        # For temporal stacking, we use a deque with fixed length.
        self.state_history = deque(maxlen=STACK_SIZE)
        self.reset()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = (1, 0)  # Start moving right.
        self.place_food()
        self.game_over = False
        self.score = 0
        self.move_counter = 0
        # Initialize the "last rewarded tile" using the Euclidean distance.
        self.last_rewarded_tile = int(self.euclidean_distance(self.snake[0], self.food))
        # Clear and initialize state history with the initial state.
        self.state_history.clear()
        initial_state = self.get_single_state()
        for _ in range(STACK_SIZE):
            self.state_history.append(initial_state)
        return self.get_state()

    def place_food(self):
        while True:
            self.food = (random.randint(0, self.width - 1),
                         random.randint(0, self.height - 1))
            if self.food not in self.snake:
                break

    def euclidean_distance(self, point1, point2):
        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    def get_single_state(self):
        state = np.zeros((self.height, self.width), dtype=np.float32)
        for x, y in self.snake:
            state[y, x] = 1.0
        food_x, food_y = self.food
        state[food_y, food_x] = 2.0
        return state

    def get_state(self):
        """
        Return a stacked state (a numpy array of shape (STACK_SIZE, height, width)).
        """
        return np.array(self.state_history)

    def update_state_history(self, new_state):
        self.state_history.append(new_state)

    def step(self, action):
        self.move_counter += 1
        old_head = self.snake[0]
        old_distance = self.euclidean_distance(old_head, self.food)
        
        self.update_direction(action)
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Check for collision.
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake):
            self.game_over = True
            reward = -8
            self.update_state_history(self.get_single_state())
            return self.get_state(), reward, True

        self.snake.insert(0, new_head)

        if new_head == self.food:
            reward = 90  # Large positive reward for eating.
            self.score += 1
            self.place_food()
            self.last_rewarded_tile = int(self.euclidean_distance(self.snake[0], self.food))
            self.move_counter = 0
        else:
            reward = -0.02  # Base time penalty.
            self.snake.pop()
            new_distance = self.euclidean_distance(new_head, self.food)
            new_tile = int(new_distance)
            if new_tile < self.last_rewarded_tile:
                reward += 0.5  # Bonus for moving into a closer tile.
                self.last_rewarded_tile = new_tile
            else:
                reward -= 0.2  # Penalty for not improving.
            
            if self.move_counter >= MOVE_LIMIT:
                self.game_over = True
                reward = -8  # Heavy penalty for stalling.
                self.update_state_history(self.get_single_state())
                return self.get_state(), reward, True

        new_state = self.get_single_state()
        self.update_state_history(new_state)
        return self.get_state(), reward, False

    def update_direction(self, action):
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        curr_idx = directions.index(self.direction)
        if action == 1:
            new_idx = (curr_idx + 1) % 4
        elif action == 2:
            new_idx = (curr_idx - 1) % 4
        else:
            new_idx = curr_idx
        self.direction = directions[new_idx]

    def render(self, screen, clock):
        screen.fill((0, 0, 0))
        food_rect = pygame.Rect(self.food[0] * BLOCK_SIZE,
                                self.food[1] * BLOCK_SIZE,
                                BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(screen, (255, 0, 0), food_rect)
        for block in self.snake:
            block_rect = pygame.Rect(block[0] * BLOCK_SIZE,
                                     block[1] * BLOCK_SIZE,
                                     BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, (0, 255, 0), block_rect)
        pygame.display.flip()
        clock.tick(FPS)

# --------------------------
# Convolutional Neural Network Model using PyTorch
# --------------------------
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        input_shape: (channels, height, width) where channels = STACK_SIZE.
        num_actions: number of possible actions (3)
        """
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        conv_out_size = 32 * input_shape[1] * input_shape[2]
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --------------------------
# DQN Agent Implementation
# --------------------------
class DQNAgent:
    def __init__(self, input_shape, action_size):
        self.input_shape = input_shape  # (channels, height, width)
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE

        self.model = DQN(input_shape, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values, dim=1).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in minibatch]).to(device)
        actions = torch.LongTensor([e[1] for e in minibatch]).unsqueeze(1).to(device)
        rewards = torch.FloatTensor([e[2] for e in minibatch]).to(device)
        next_states = torch.FloatTensor([e[3] for e in minibatch]).to(device)
        dones = torch.FloatTensor([1 if e[4] else 0 for e in minibatch]).to(device)

        q_values = self.model(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            max_next_q_values = self.model(next_states).max(1)[0]
        targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=device))
        self.model.eval()

# --------------------------
# Training Loop
# --------------------------
def train_agent():
    game = SnakeGame()
    input_shape = (STACK_SIZE, game.height, game.width)
    action_size = 3
    agent = DQNAgent(input_shape, action_size)

    if LOAD_PRETRAINED:
        try:
            agent.load(MODEL_FILE_TRAIN)
            print("Loaded pre-trained weights.")
        except Exception as e:
            print("Could not load pre-trained weights. Starting fresh.")

    scores = []
    moving_avg_scores = []

    for episode in range(1, NUM_EPISODES + 1):
        state = game.reset()  # state shape: (STACK_SIZE, H, W)
        total_reward = 0
        step_count = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            agent.replay(BATCH_SIZE)
            step_count += 1
            if done:
                break

        scores.append(game.score)
        moving_avg = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
        moving_avg_scores.append(moving_avg)
        print(f"Episode {episode}/{NUM_EPISODES} - Score: {game.score} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f} - Moving Avg Score: {moving_avg:.2f}")

        # Use a try/except block to write to the stats file
        try:
            with open(STATS_FILE, "a") as f:
                f.write(f"{episode},{moving_avg:.2f}\n")
        except PermissionError:
            print(f"PermissionError: Unable to write to {STATS_FILE}. Trying a new file...")
            new_stats_file = get_stats_filename(BASE_STATS_FILE)
            with open(new_stats_file, "a") as f:
                f.write(f"{episode},{moving_avg:.2f}\n")
            STATS_FILE = new_stats_file

        if episode % 100 == 0:
            model_filename = f"{episode}_{MODEL_FILE}"
            agent.save(model_filename)
            print(f"Model weights saved at episode {episode}.")

        if episode % 100 == 0:
            if moving_avg < 0.5 and agent.epsilon < 0.35:
                agent.epsilon = 0.5
                print("Moving average is low and epsilon is low; resetting epsilon to 0.5 for additional exploration.")

    agent.save(MODEL_FILE)
    print("Training completed and final model saved.")
# --------------------------
# Play Mode: Visualize the Trained Agent
# --------------------------
def play_agent():
    pygame.init()
    screen = pygame.display.set_mode((GAME_WIDTH * BLOCK_SIZE, GAME_HEIGHT * BLOCK_SIZE))
    pygame.display.set_caption("Snake RL Agent - Play Mode")
    clock = pygame.time.Clock()

    game = SnakeGame()
    input_shape = (STACK_SIZE, game.height, game.width)
    action_size = 3
    agent = DQNAgent(input_shape, action_size)
    try:
        agent.load(MODEL_FILE)
        print("Loaded trained model weights.")
    except Exception as e:
        print("Error loading model weights. Train the model first.")
        sys.exit()

    state = game.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # shape: (1, STACK_SIZE, H, W)
        with torch.no_grad():
            q_values = agent.model(state_tensor)
        action = torch.argmax(q_values, dim=1).item()

        next_state, reward, done = game.step(action)
        state = next_state

        game.render(screen, clock)
        if done:
            print(f"Game over! Score: {game.score}")
            time.sleep(2)
            state = game.reset()

    pygame.quit()

# --------------------------
# Main Entry Point
# --------------------------
if __name__ == '__main__':
    if TRAIN_MODE:
        print("Starting training mode...")
        train_agent()
    else:
        print("Starting play mode...")
        play_agent()
