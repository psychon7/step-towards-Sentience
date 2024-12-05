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
global_workspace_memory = deque(maxlen=10000)  # Increased memory
emotional_state = {'curiosity': 0.7, 'fear': 0.1, 'reward_seeking': 0.2}  # Adjusted initial emotions

# Initialize Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=2):  # Larger network
        super(ImprovedNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout for better generalization
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def update_emotional_state(reward):
    global emotional_state
    if reward > 0:
        emotional_state['reward_seeking'] += 0.1  # Increased positive reinforcement
        emotional_state['fear'] = max(0, emotional_state['fear'] - 0.05)
        emotional_state['curiosity'] = min(1.0, emotional_state['curiosity'] + 0.02)
    else:
        emotional_state['fear'] += 0.02  # Reduced negative impact
        emotional_state['reward_seeking'] = max(0, emotional_state['reward_seeking'] - 0.01)
    
    # Normalize
    total = sum(emotional_state.values())
    for key in emotional_state:
        emotional_state[key] /= total

def train():
    env = gym.make('CartPole-v1', render_mode='human')
    initial_obs, _ = env.reset()
    env.render()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedNN().to(device)
    target_model = ImprovedNN().to(device)  # Added target network
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Reduced learning rate
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer()
    
    # Improved training parameters
    num_episodes = 500  # More episodes
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    epsilon = epsilon_start
    batch_size = 64
    target_update = 10  # Update target network every N episodes
    min_replay_size = 1000  # Start training after this many experiences
    
    try:
        for episode in range(num_episodes):
            state = torch.FloatTensor(initial_obs).unsqueeze(0).to(device)
            total_reward = 0
            step = 0
            done = False
            
            while not done:
                env.render()
                
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = model(state)
                        action = q_values.argmax().item()
                
                # Take action
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Modified reward based on emotional state
                modified_reward = (reward * emotional_state['reward_seeking'] + 
                                 reward * emotional_state['curiosity'] - 
                                 abs(reward) * emotional_state['fear'])
                
                # Store experience
                next_state = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
                replay_buffer.push(state, action, modified_reward, next_state, done)
                
                # Train on batch if enough experiences
                if len(replay_buffer) > min_replay_size:
                    batch = replay_buffer.sample(batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)
                    
                    states = torch.cat(states)
                    next_states = torch.cat(next_states)
                    actions = torch.tensor(actions, device=device)
                    rewards = torch.tensor(rewards, device=device)
                    dones = torch.tensor(dones, dtype=torch.float32, device=device)
                    
                    # Compute current Q values
                    current_q = model(states).gather(1, actions.unsqueeze(1))
                    
                    # Compute next Q values using target network
                    with torch.no_grad():
                        next_q = target_model(next_states).max(1)[0]
                        target_q = rewards + (0.99 * next_q * (1 - dones))
                    
                    # Update model
                    loss = criterion(current_q.squeeze(), target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Added gradient clipping
                    optimizer.step()
                
                state = next_state
                total_reward += reward
                step += 1
                update_emotional_state(modified_reward)
                
                time.sleep(0.01)  # Reduced delay for faster training
            
            # Update target network
            if episode % target_update == 0:
                target_model.load_state_dict(model.state_dict())
            
            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            logger.info(f"Episode {episode + 1} - Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
            
            # Early stopping if solved
            if total_reward > 475:  # Increased threshold
                logger.info(f"Environment solved in {episode + 1} episodes!")
                break
                
            initial_obs, _ = env.reset()
                
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    finally:
        env.close()
        logger.info("Training completed")

if __name__ == "__main__":
    train()