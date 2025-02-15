Deep Q-Learning for Snake AI with PyTorch and CUDA
Project Overview

This project aims to train an AI agent to play the classic Snake game using Deep Q-Learning (DQN) with PyTorch and CUDA acceleration. The agent learns through reinforcement learning by interacting with the environment and optimizing a neural network to maximize its score.

As of now, the project is a work in progress and has not achieved full success. However, some progress has been made, with the agent reaching a moving average score of ~1.7 cherries per game after 45,000 games.
Motivation and Design Choices
1. Reinforcement Learning Algorithm

    DQN (Deep Q-Network)
        Chosen due to its success in learning game strategies from raw pixel input.
        Uses a neural network to approximate the Q-value function.
        Experience replay is used to break correlation between consecutive training samples.

2. Environment: Snake Game

    Built using PyGame for rendering and game logic.
    Grid-based movement system where the state consists of:
        The snake's position.
        The food’s position.
        The snake's direction.
    Reward system:
        +90 reward for eating a cherry.
        -0.02 per move (time penalty).
        -0.2 for not moving towards food.
        Termination if the snake does not eat within 200 moves.

3. Neural Network Architecture

    A convolutional neural network (CNN) processes the game state as a (3, 20, 20) tensor.
    The network consists of:
        Two convolutional layers to extract spatial features.
        Fully connected layers to map features to Q-values for each possible action.

4. Temporal Stacking of States

    Instead of using a single frame, the model stacks the last 3 frames.
    This allows the model to infer motion, helping it understand the direction of movement.

5. Action Space

    3 possible actions:
        Turn Left
        Turn Right
        Keep moving forward
    The agent is not allowed to move backward to prevent self-collisions.

6. Experience Replay

    Implemented using a deque (FIFO memory buffer) with a size of 2000 experiences.
    Mini-batches of 64 samples are randomly selected for training.
    This helps break correlations between consecutive frames, improving learning stability.

7. Epsilon-Greedy Exploration Strategy

    Epsilon-Decay:
        Starts at 0.3.
        Gradually decreases to 0.01 over time.
        Every 100 episodes, if performance is low, epsilon is temporarily reset to 0.5 to encourage exploration.

8. Model Training and Logging

    Training runs for 50,000 games.
    Moving average score is logged every episode.
    Model is saved every 100 episodes to allow resuming training later.

Current Results

    After 45,000 games, the AI reaches an average score of 1.7 cherries per game.
    Learning is slow, suggesting potential issues:
        Reward shaping may need further tuning.
        Network architecture may require adjustments.
        Exploration-exploitation balance might need tweaking.
![Figure_1](https://github.com/user-attachments/assets/f6101c4b-0c4d-4018-bbf5-b2e6c51aad16)

Future Improvements

    Refine reward function to better incentivize snake behavior.
    Optimize CNN architecture to improve feature extraction.
    Experiment with different hyperparameters, such as learning rate and memory size.
    Implement Double DQN (DDQN) to reduce overestimation of Q-values.
    Try different action spaces, such as allowing finer movement control.

Installation and Running Instructions
1. Install Dependencies

pip install torch pygame numpy

2. Train the Model

python snake_rl.py

3. Play with a Trained Model

python snake_rl.py --play

Conclusion

This project is an ongoing experiment in reinforcement learning applied to the Snake game. While the model is not fully optimized yet, it has shown some progress. Future iterations will focus on improving efficiency, rewards, and exploration strategies.

Update 1:

In an attempt to increase improvement rates a DuelQN algortihm was implemented and code posted.
