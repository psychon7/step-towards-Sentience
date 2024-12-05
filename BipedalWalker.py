import numpy as np
import random
import gym
import logging
from collections import deque
import norse.torch as norse
import torch
import torch.nn as nn
import torch.optim as optim
import cv2  # OpenCV for vision input simulation

# Global Parameters
global_workspace_memory = deque(maxlen=2000)  # Increased memory to hold more significant experiences
long_term_memory = []  # Long-term memory to hold important experiences
associative_memory = {}  # Associative memory for concept relations
emotional_state = {'curiosity': 0.5, 'fear': 0.1, 'reward_seeking': 0.3}

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Emotional State Update Function
def update_emotional_state(reward):
    if reward > 0:
        emotional_state['reward_seeking'] += 0.01
        emotional_state['fear'] = max(0, emotional_state['fear'] - 0.005)
    else:
        emotional_state['fear'] += 0.02
        emotional_state['reward_seeking'] = max(0, emotional_state['reward_seeking'] - 0.01)
    # Normalize to keep values between 0 and 1
    total = sum(emotional_state.values())
    for key in emotional_state:
        emotional_state[key] /= total

# Metacognitive Feedback - Self Evaluation
def metacognitive_feedback(action, reward, step):
    feedback = f"Step {step} - Action: {action}, Reward: {reward}, Emotional State: {emotional_state}"
    logger.info(feedback)
    global_workspace_memory.append(feedback)
    # Store significant experience in long-term memory
    if reward > 1.0 or reward < -1.0:
        long_term_memory.append((action, reward, emotional_state.copy()))

# Associative Memory Update Function
def update_associative_memory(state, action, reward):
    key = (tuple(state.tolist()), action)
    if key in associative_memory:
        associative_memory[key].append(reward)
    else:
        associative_memory[key] = [reward]

# Spiking Neural Network with Hierarchical Model (Norse + LSTM + Dynamic Expansion)
class AdvancedNN(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, output_size=4):
        super(AdvancedNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(32 * 6 * 6, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + input_size, output_size)
        )
        self.spiking_layer = norse.torch.LIFRecurrentCell(output_size, output_size)

    def forward(self, image, state, hidden):
        x = self.cnn(image)
        lstm_out, hidden = self.lstm(x.unsqueeze(0), hidden)
        combined = torch.cat((lstm_out[:, -1, :], state), dim=1)
        output = self.fc(combined)
        z, s1 = self.spiking_layer(output)
        return z, s1, hidden

    def expand_network(self):
        # Dynamically expand network complexity (Recursive Self-Improvement)
        new_layer = nn.Linear(self.fc[-1].out_features, self.fc[-1].out_features)
        self.fc = nn.Sequential(
            *list(self.fc.children()),
            nn.ReLU(),
            new_layer
        )
        logger.info("Expanded network architecture for enhanced complexity.")

# Environment Setup (Gym Environment)
env = gym.make('BipedalWalker-v3')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

# Initialize Advanced Neural Network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdvancedNN(input_size=state_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Parameters for Learning
num_episodes = 500
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995  # Decay rate for epsilon
epsilon_min = 0.01  # Minimum epsilon value
hidden = (torch.zeros(1, 1, 128).to(device), torch.zeros(1, 1, 128).to(device))

# Main Training Loop
for episode in range(num_episodes):
    state = torch.FloatTensor(env.reset()).unsqueeze(0).to(device)
    total_reward = 0
    step = 0
    done = False

    logger.info(f"\nStarting Episode {episode + 1}")

    while not done:
        # Simulate Visual Input (Randomly generated for example purposes)
        visual_obs = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
        visual_obs = torch.FloatTensor(visual_obs).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

        # Choose action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action, _, _ = model(visual_obs, state, hidden)
                action = action.squeeze().cpu().numpy()

        # Take action
        next_obs, reward, done, _ = env.step(action)
        next_state = torch.FloatTensor(next_obs).unsqueeze(0).to(device)

        # Emotional Influence on Reward
        modified_reward = reward * emotional_state['reward_seeking'] - emotional_state['fear']

        # Forward pass and calculate target
        current_q, _, hidden = model(visual_obs, state, hidden)
        next_q, _, _ = model(visual_obs, next_state, hidden)
        target_q = current_q.clone()
        target_q[0, action.argmax()] = modified_reward + (0.99 * next_q.max(1)[0] * (not done))

        # Update model
        loss = criterion(current_q, target_q.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update emotional state and provide feedback
        update_emotional_state(modified_reward)
        metacognitive_feedback(action, modified_reward, step)
        update_associative_memory(state.cpu().numpy(), action.argmax(), reward)

        # Recursive Self-Improvement Condition
        if len(long_term_memory) > 100 and random.random() < 0.01:
            model.expand_network()

        # Move to next state
        state = next_state
        total_reward += reward
        step += 1

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    logger.info(f"Episode {episode + 1} finished with total reward: {total_reward:.2f}")

    # Early stopping if we solve the environment
    if total_reward > 195.0:
        logger.info(f"Environment solved in {episode + 1} episodes!")
        break

env.close()
