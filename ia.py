import numpy as np
import random
import gym
import logging
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Global Parameters
global_workspace_memory = deque(maxlen=100)
emotional_state = {'curiosity': 0.5, 'fear': 0.1, 'reward_seeking': 0.3}

# Initialize Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdaptiveNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=2):
        super(AdaptiveNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

def update_emotional_state(reward):
    if reward > 0:
        emotional_state['reward_seeking'] += 0.05
        emotional_state['fear'] = max(0, emotional_state['fear'] - 0.02)
    else:
        emotional_state['fear'] += 0.05
        emotional_state['reward_seeking'] = max(0, emotional_state['reward_seeking'] - 0.02)
    total = sum(emotional_state.values())
    for key in emotional_state:
        emotional_state[key] /= total

def metacognitive_feedback(action, reward, step):
    feedback = f"Step {step} - Action: {action}, Reward: {reward:.2f}, Emotional State: {emotional_state}"
    logger.info(feedback)
    global_workspace_memory.append(feedback)

def train():
    # Environment setup
    env = gym.make('CartPole-v1', render_mode=None)
    initial_obs, _ = env.reset(seed=42)
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptiveNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training parameters
    num_episodes = 100
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    epsilon = epsilon_start
    
    try:
        for episode in range(num_episodes):
            state = torch.FloatTensor(initial_obs).unsqueeze(0).to(device)
            total_reward = 0
            step = 0
            done = False
            
            logger.info(f"\nStarting Episode {episode + 1}")
            
            while not done:
                # Choose action
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = model(state).argmax().item()
                
                # Take action
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Apply emotional influence
                modified_reward = reward * emotional_state['reward_seeking'] - emotional_state['fear']
                
                # Convert to tensor
                next_state = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
                reward_tensor = torch.tensor([modified_reward], device=device)
                
                # Learn
                current_q = model(state)
                next_q = model(next_state)
                target_q = current_q.clone()
                target_q[0, action] = reward_tensor + (0.99 * next_q.max(1)[0] * (not done))
                
                # Update model
                loss = criterion(current_q, target_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update state and tracking
                state = next_state
                total_reward += reward
                step += 1
                
                # Update emotional state and provide feedback
                update_emotional_state(modified_reward)
                metacognitive_feedback(action, modified_reward, step)
                
                # Small delay to make visualization viewable
                time.sleep(0.01)
            
            # Episode cleanup
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            logger.info(f"Episode {episode + 1} finished with total reward: {total_reward:.2f}")
            initial_obs, _ = env.reset()
            
            # Early stopping if we solve the environment
            if total_reward > 195.0:
                logger.info(f"Environment solved in {episode + 1} episodes!")
                break
                
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    finally:
        env.close()
        logger.info("Training completed")

if __name__ == "__main__":
    train()