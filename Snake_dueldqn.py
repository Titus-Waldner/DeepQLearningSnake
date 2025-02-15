"""
Snake RL with Dueling DQN (PyTorch and PyGame)

Features in this version:
 - Uses a dueling DQN architecture with separate advantage and value streams.
 - Maintains a target network (updated every TARGET_UPDATE episodes) for more stable learning.
 - Uses a reward function based on change in Euclidean distance with additional time, food, and death penalties.
 - Stacks the last few states (temporal frames) for the network.
 - Saves model weights every SAVE_INTERVAL episodes.
 - Includes an epsilon reset mechanic to boost exploration if performance stagnates.
 
Before running:
    pip install torch pygame numpy
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
# Hyperparameters and Parameters
# --------------------------
# Game parameters
BLOCK_SIZE = 20           # Pixel size of each block
GAME_WIDTH = 20           # Number of blocks horizontally
GAME_HEIGHT = 20          # Number of blocks vertically
FPS = 60                  # For visualization

# DQN parameters
MEMORY_SIZE = 40000       # Replay memory capacity
BATCH_SIZE = 64           # Mini-batch size for training
GAMMA = 0.99              # Discount factor
LEARNING_RATE = 1e-4      # Learning rate for the optimizer
EPSILON_START = 0.3       # Initial exploration rate
EPSILON_MIN = 0.01        # Minimum exploration rate
EPSILON_DECAY = 0.9995    # Slower decay factor per episode (10x slower than 0.995)
TARGET_UPDATE = 10        # Update target network every N episodes

NUM_EPISODES = 10000      # Total training episodes
MOVE_LIMIT = 200          # Maximum moves per episode

# Temporal stacking of states (channels)
STACK_SIZE = 3

# Save & load settings
TRAIN_MODE = True         # True: training; False: play mode
LOAD_PRETRAINED = False   # Change to True if you want to load a saved model
MODEL_FILE = "10000_snake_dueling_dqn"
SAVE_INTERVAL = 100       # Save model every SAVE_INTERVAL episodes

# --------------------------
# Utility for Logging Stats
# --------------------------
def get_stats_filename(base_name="stats.txt"):
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

STATS_FILE = get_stats_filename()

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
        self.state_history = deque(maxlen=STACK_SIZE)
        self.reset()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = (1, 0)  # Start moving right.
        self.place_food()
        self.game_over = False
        self.score = 0
        self.move_counter = 0
        # For reward shaping: track the previous Euclidean distance from head to food.
        self.prev_distance = self.euclidean_distance(self.snake[0], self.food)
        # Initialize the state history with the initial state.
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
        # Mark snake’s body with 1.0
        for x, y in self.snake:
            state[y, x] = 1.0
        # Mark food with 2.0
        food_x, food_y = self.food
        state[food_y, food_x] = 2.0
        return state

    def get_state(self):
        """
        Returns a stacked state with shape (STACK_SIZE, height, width)
        """
        return np.array(self.state_history)

    def update_state_history(self, new_state):
        self.state_history.append(new_state)

    def step(self, action):
        """
        Actions: 0 - continue straight, 1 - turn right, 2 - turn left
        """
        self.move_counter += 1
        old_head = self.snake[0]
        old_distance = self.euclidean_distance(old_head, self.food)

        self.update_direction(action)
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Check for collision (with wall or self)
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake):
            self.game_over = True
            reward = -10  # Heavy penalty for collision
            self.update_state_history(self.get_single_state())
            return self.get_state(), reward, True

        self.snake.insert(0, new_head)
        new_distance = self.euclidean_distance(new_head, self.food)

        # If food is eaten
        if new_head == self.food:
            reward = 10  # Large reward for eating
            self.score += 1
            self.place_food()
            # Reset move counter and distance after eating
            self.move_counter = 0
            new_distance = self.euclidean_distance(new_head, self.food)
        else:
            # Remove tail if no food is eaten
            self.snake.pop()
            # Reward shaping: small reward for moving closer, penalty for moving away.
            reward = -0.01 + 0.1 * (old_distance - new_distance)

            # If too many moves without eating, end the episode.
            if self.move_counter >= MOVE_LIMIT:
                self.game_over = True
                reward = -10

        # Update previous distance for next step.
        self.prev_distance = new_distance
        new_state = self.get_single_state()
        self.update_state_history(new_state)
        return self.get_state(), reward, self.game_over

    def update_direction(self, action):
        # Directions: Up, Right, Down, Left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        curr_idx = directions.index(self.direction)
        if action == 1:   # Turn right
            new_idx = (curr_idx + 1) % 4
        elif action == 2: # Turn left
            new_idx = (curr_idx - 1) % 4
        else:             # Continue straight
            new_idx = curr_idx
        self.direction = directions[new_idx]

    def render(self, screen, clock):
        screen.fill((0, 0, 0))
        # Draw food in red
        food_rect = pygame.Rect(self.food[0] * BLOCK_SIZE,
                                self.food[1] * BLOCK_SIZE,
                                BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(screen, (255, 0, 0), food_rect)
        # Draw snake in green
        for block in self.snake:
            block_rect = pygame.Rect(block[0] * BLOCK_SIZE,
                                     block[1] * BLOCK_SIZE,
                                     BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, (0, 255, 0), block_rect)
        pygame.display.flip()
        clock.tick(FPS)

# --------------------------
# Dueling DQN Model Definition
# --------------------------
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        input_shape: (channels, height, width)
        num_actions: number of possible actions (3)
        """
        super(DuelingDQN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # Compute the size of the flattened features
        conv_out_size = 64 * input_shape[1] * input_shape[2]
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine value and advantage streams into Q-values
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals

# --------------------------
# Dueling DQN Agent with Target Network
# --------------------------
class DQNAgent:
    def __init__(self, input_shape, action_size):
        self.input_shape = input_shape  # e.g., (STACK_SIZE, H, W)
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE

        # Online network and target network (for stability)
        self.model = DuelingDQN(input_shape, action_size).to(device)
        self.target_model = DuelingDQN(input_shape, action_size).to(device)
        self.update_target_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # shape: (1, channels, H, W)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values, dim=1).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        # Convert minibatch to tensors
        states = torch.FloatTensor([e[0] for e in minibatch]).to(device)
        actions = torch.LongTensor([e[1] for e in minibatch]).unsqueeze(1).to(device)
        rewards = torch.FloatTensor([e[2] for e in minibatch]).to(device)
        next_states = torch.FloatTensor([e[3] for e in minibatch]).to(device)
        dones = torch.FloatTensor([1 if e[4] else 0 for e in minibatch]).to(device)

        # Current Q-values from online network
        q_values = self.model(states).gather(1, actions).squeeze(1)
        # Use the target network to get the next Q-values
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1)[0]
        # Compute targets
        targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=device))
        self.update_target_network()

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
            agent.load(MODEL_FILE)
            print("Loaded pre-trained weights.")
        except Exception as e:
            print("Could not load pre-trained weights. Starting fresh.")

    scores = []
    moving_avg_scores = []

    for episode in range(1, NUM_EPISODES + 1):
        state = game.reset()  # shape: (STACK_SIZE, H, W)
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            agent.replay(BATCH_SIZE)
            if done:
                break

        # Decay epsilon after each episode
        agent.decay_epsilon()
        scores.append(game.score)
        moving_avg = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
        moving_avg_scores.append(moving_avg)
        print(f"Episode {episode}/{NUM_EPISODES} - Score: {game.score} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f} - Moving Avg Score: {moving_avg:.2f}")

        # Log stats to file
        try:
            with open(STATS_FILE, "a") as f:
                f.write(f"{episode},{moving_avg:.2f}\n")
        except PermissionError:
            print(f"PermissionError: Unable to write to {STATS_FILE}. Logging skipped for this episode.")

        # Update target network periodically
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
            print("Target network updated.")

        # Save the model periodically
        if episode % SAVE_INTERVAL == 0:
            model_filename = f"{episode}_{MODEL_FILE}"
            agent.save(model_filename)
            print(f"Model saved at episode {episode}.")

        # Reset epsilon if performance is poor and epsilon is too low
        if episode % 100 == 0:
            # If the moving average score is very low (e.g., below 0.5) and epsilon is nearly minimal, reset to 0.5.
            if moving_avg < 0.5 and agent.epsilon < 0.35:
                agent.epsilon = 0.5
                print("Poor performance detected. Resetting epsilon to 0.5 for additional exploration.")

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
        # Process quit events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
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
