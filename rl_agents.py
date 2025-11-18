"""
Hierarchical Multi-Agent RL System for PU Optimization
"""
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
import os
import pickle


class SimplePolicyNetwork(nn.Module):
    """Simple policy network for RL agents - optimized for speed"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=32):  # Reduced from 64 to 32
        super(SimplePolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)  # Removed one layer
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.fc2(x)  # Direct to output
        return self.softmax(x)


class BaseAgent:
    """Base class for RL agents"""
    
    def __init__(self, state_dim, action_dim, learning_rate=0.003, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        self.policy_net = SimplePolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def select_action(self, state, available_actions=None, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            if available_actions:
                return random.choice(available_actions)
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state_tensor)
        
        if available_actions:
            # Mask unavailable actions
            mask = torch.zeros(self.action_dim)
            for action in available_actions:
                mask[action] = 1.0
            probs = probs * mask
            probs = probs / probs.sum()
        
        action = torch.multinomial(probs, 1).item()
        return action
    
    def update(self, states, actions, rewards):
        """Update policy using REINFORCE algorithm"""
        if len(states) == 0:
            return
        
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        
        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns_tensor = torch.FloatTensor(returns)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # Get action probabilities
        probs = self.policy_net(states_tensor)
        log_probs = torch.log(probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1) + 1e-8)
        
        # Policy gradient loss
        loss = -(log_probs * returns_tensor).mean()
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save agent model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, filepath)
    
    def load_model(self, filepath):
        """Load agent model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            return True
        return False
    
    def set_eval_mode(self):
        """Set agent to evaluation mode (no exploration)"""
        self.policy_net.eval()
        self.epsilon = 0.0  # No exploration in eval mode
    
    def set_train_mode(self):
        """Set agent to training mode"""
        self.policy_net.train()


class ManagerAgent(BaseAgent):
    """
    Manager Agent: Oversees rules and PU usage constraints
    - Maximum PU usage limits
    - Fresh PU assignment rules
    - Coordinates with co-worker agents
    """
    
    def __init__(self, state_dim, action_dim, learning_rate=0.003, gamma=0.99):
        super().__init__(state_dim, action_dim, learning_rate, gamma)
        self.max_pu_usage = 3  # Maximum number of PUs that can be used
        self.performance_weight = 0.3
        self.reliability_weight = 0.3
        
    def calculate_reward(self, state, action, next_state, info, performance_reward, reliability_reward):
        """Calculate reward based on rule compliance"""
        reward = 0.0
        
        # Check maximum PU usage constraint
        # This is handled implicitly by available actions
        
        # Reward for following Fresh PU constraints
        fresh_pu_constraints = state[16:19]  # Fresh PU indicators
        if np.any(fresh_pu_constraints > 0.5):
            # A fresh PU is required
            required_pu = np.argmax(fresh_pu_constraints)
            if action == required_pu:
                reward += 10.0  # Strong reward for following constraint
            else:
                reward -= 20.0  # Strong penalty for violating constraint
        
        # Coordinate with co-workers: incorporate their rewards
        reward += self.performance_weight * performance_reward + self.reliability_weight * reliability_reward
        
        # RELIABILITY PRIORITY: If reliability agent gives very negative reward, prioritize it
        if reliability_reward < -100.0:  # Critical reliability issue
            reward += reliability_reward * 2.0  # Double the reliability penalty
        
        # Penalty for invalid actions (should be handled by action masking)
        available_mask = state[13:16]  # Available PU mask
        if available_mask[action] < 0.5:
            reward -= 50.0
        
        # Additional check: prevent selecting PU with critically low RUL
        rul = state[1:4]
        selected_rul = rul[action]
        if selected_rul < 0.05:  # Less than 5% RUL
            reward -= 300.0  # Severe penalty
        elif selected_rul < 0.1:  # Less than 10% RUL
            reward -= 100.0  # Strong penalty
        
        return reward
    
    def update_reward_weights(self, performance_weight=0.3, reliability_weight=0.3):
        """Update reward weights for manager agent based on bias"""
        # This can be used to adjust how manager weighs co-worker recommendations
        # Stored for use in calculate_reward
        self.performance_weight = performance_weight
        self.reliability_weight = reliability_weight


class PerformanceAgent(BaseAgent):
    """
    Performance Co-worker Agent: Minimizes performance degradation
    - Focuses on maintaining maximum power throughout season
    - Minimizes cumulative power reduction
    """
    
    def __init__(self, state_dim, action_dim, learning_rate=0.003, gamma=0.99):
        super().__init__(state_dim, action_dim, learning_rate, gamma)
    
    def calculate_reward(self, state, action, next_state, info, damage_model_func, current_solution):
        """Calculate reward based on performance metrics"""
        reward = 0.0
        
        # Get current power states
        power_left = state[4:7]  # Power left for each PU (normalized)
        power_reduced = state[7:10]  # Cumulative power reduced (normalized)
        
        # Reward for selecting PU with higher remaining power
        selected_pu_power = power_left[action]
        reward += selected_pu_power * 5.0
        
        # Penalty for increasing power degradation
        # Lower cumulative power reduction is better
        avg_power_reduced = np.mean(power_reduced)
        reward -= avg_power_reduced * 10.0
        
        # Calculate actual power loss from damage model
        try:
            _, PowerLoss, _, _, _ = damage_model_func(current_solution)
            # Negative power loss is good (less degradation)
            reward += (-PowerLoss / 100.0)  # Normalize
        except:
            pass
        
        return reward


class ReliabilityAgent(BaseAgent):
    """
    Reliability Co-worker Agent: Ensures SOH (RUL) > 0 until end of season
    - Monitors RUL for each PU
    - Ensures PUs remain alive until end of season
    - Balances PU usage to prevent early failures
    """
    
    def __init__(self, state_dim, action_dim, learning_rate=0.003, gamma=0.99):
        super().__init__(state_dim, action_dim, learning_rate, gamma)
    
    def calculate_reward(self, state, action, next_state, info, damage_model_func, current_solution):
        """Calculate reward based on reliability metrics - STRONG constraints to prevent failures"""
        reward = 0.0
        
        # Get current RUL states (normalized, 0-1 scale where 1.0 = 100 RUL)
        rul = state[1:4]  # RUL for each PU (normalized)
        selected_pu_rul = rul[action]
        min_rul = np.min(rul)
        
        # Calculate actual RUL from damage model for final validation
        try:
            _, _, _, RUL_df, _ = damage_model_func(current_solution)
            
            # CRITICAL: Check if any race has negative RUL (PU failure) - EXTREME penalty
            failed_races = RUL_df[RUL_df["RUL"] < 0]
            if len(failed_races) > 0:
                reward -= 2000.0  # EXTREME penalty for any failure
            
            # Get minimum RUL across all races (worst case)
            min_rul_actual = RUL_df["RUL"].min() if len(RUL_df) > 0 else 100.0
            
            # Convert normalized RUL to actual (assuming max is 100)
            selected_pu_rul_actual = selected_pu_rul * 100.0
            min_rul_actual_from_state = min_rul * 100.0
            
            # CRITICAL: Prevent using PU with very low RUL
            if selected_pu_rul_actual < 5.0:  # Less than 5% RUL
                reward -= 800.0  # Very severe penalty
            elif selected_pu_rul_actual < 10.0:  # Less than 10% RUL
                reward -= 400.0  # Severe penalty
            elif selected_pu_rul_actual < 20.0:  # Less than 20% RUL
                reward -= 100.0  # Moderate penalty
            
            # Check overall system health
            if min_rul_actual < 0:
                reward -= 2000.0  # EXTREME penalty
            elif min_rul_actual < 5.0:  # Any PU very close to failure
                reward -= 600.0  # Severe warning
            elif min_rul_actual < 10.0:
                reward -= 300.0  # Warning
            elif min_rul_actual < 20.0:
                reward -= 100.0  # Caution
            
            # Reward for selecting PU with higher RUL
            reward += selected_pu_rul * 30.0
            
            # Reward for balanced RUL across PUs (prevent overusing one PU)
            rul_std = np.std(rul)
            reward -= rul_std * 15.0  # Stronger penalty for imbalance
            
            # Bonus for keeping all PUs above safe threshold
            if min_rul_actual > 20.0 and min_rul > 0.2:
                reward += 100.0  # Bonus for safe operation
        except:
            # Fallback to state-based rewards (use during fast training)
            # CRITICAL: Prevent failures using state RUL
            if min_rul < 0:
                reward -= 2000.0  # EXTREME penalty
            elif min_rul < 0.05:  # Less than 5% RUL (critical)
                reward -= 800.0  # Very severe penalty
            elif min_rul < 0.1:  # Less than 10% RUL (warning)
                reward -= 400.0  # Severe penalty
            elif min_rul < 0.2:  # Less than 20% RUL
                reward -= 100.0  # Moderate penalty
            
            # Prevent using PU with low RUL
            if selected_pu_rul < 0.05:
                reward -= 800.0
            elif selected_pu_rul < 0.1:
                reward -= 400.0
            elif selected_pu_rul < 0.2:
                reward -= 100.0
            
            # Reward for higher RUL
            reward += selected_pu_rul * 30.0
            
            # Penalty for imbalance
            rul_std = np.std(rul)
            reward -= rul_std * 15.0
        
        return reward


class HierarchicalRLCoordinator:
    """
    Coordinates the hierarchical RL system with Manager and two Co-worker agents
    """
    
    def __init__(self, env, damage_model_func, num_episodes=100, use_pretrained=True):
        self.env = env
        self.damage_model_func = damage_model_func
        self.num_episodes = num_episodes
        self.use_pretrained = use_pretrained
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        # Initialize agents
        self.manager = ManagerAgent(state_dim, action_dim)
        self.performance_agent = PerformanceAgent(state_dim, action_dim)
        self.reliability_agent = ReliabilityAgent(state_dim, action_dim)
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Try to load pre-trained models
        if use_pretrained:
            self.load_models()
        
    def get_available_actions(self, state):
        """Get list of available actions based on state - with reliability constraints"""
        available_mask = state[13:16]  # Available PU mask
        available_actions = [i for i in range(3) if available_mask[i] > 0.5]
        
        # Check Fresh PU constraint
        fresh_pu_constraints = state[16:19]
        if np.any(fresh_pu_constraints > 0.5):
            required_pu = np.argmax(fresh_pu_constraints)
            if required_pu in available_actions:
                return [required_pu]  # Only this PU is allowed
        
        # RELIABILITY CONSTRAINT: Remove PUs with critically low RUL
        rul = state[1:4]  # RUL for each PU (normalized)
        
        # First, filter out PUs with very low RUL (< 10% for safety margin)
        safe_actions = []
        for action in available_actions:
            if rul[action] > 0.10:  # Only allow PUs with > 10% RUL (safety margin)
                safe_actions.append(action)
        
        # If no safe actions, relax to 5% threshold
        if len(safe_actions) == 0:
            for action in available_actions:
                if rul[action] > 0.05:  # Allow PUs with > 5% RUL
                    safe_actions.append(action)
        
        # If still no safe actions, use the one with highest RUL (emergency)
        if len(safe_actions) == 0 and len(available_actions) > 0:
            # Emergency: use the PU with highest RUL even if critical
            best_pu = max(available_actions, key=lambda a: rul[a])
            return [best_pu]
        
        return safe_actions if safe_actions else available_actions
    
    def train_episode(self, episode_num, progress_callback=None, fast_mode=True):
        """Train for one episode - optimized for speed"""
        state, info = self.env.reset()
        
        episode_states = []
        episode_actions = []
        manager_rewards = []
        performance_rewards = []
        reliability_rewards = []
        
        total_reward = 0.0
        step_count = 0
        
        # Cache for damage model (only calculate once at end)
        damage_cache = None
        
        while True:
            # Get available actions
            available_actions = self.get_available_actions(state)
            
            if not available_actions:
                break
            
            # Manager selects action (with input from co-workers)
            # In fast mode, skip co-worker recommendations during training
            if not fast_mode:
                performance_action = self.performance_agent.select_action(
                    state, available_actions, training=True
                )
                reliability_action = self.reliability_agent.select_action(
                    state, available_actions, training=True
                )
            
            # Manager makes final decision
            manager_action = self.manager.select_action(
                state, available_actions, training=True
            )
            
            # Use manager's decision
            action = manager_action
            
            # Store state and action
            episode_states.append(state)
            episode_actions.append(action)
            
            # Take step in environment
            next_state, _, terminated, truncated, info = self.env.step(action)
            
            # Fast mode: calculate simple rewards without expensive damage model
            if fast_mode:
                # Simple state-based rewards (much faster)
                perf_reward = self._fast_performance_reward(state, action)
                rel_reward = self._fast_reliability_reward(state, action)
                mgr_reward = self._fast_manager_reward(state, action, perf_reward, rel_reward)
            else:
                # Full reward calculation (slower but more accurate)
                current_solution = self.env.get_final_solution()
                perf_reward = self.performance_agent.calculate_reward(
                    state, action, next_state, info, self.damage_model_func, current_solution
                )
                rel_reward = self.reliability_agent.calculate_reward(
                    state, action, next_state, info, self.damage_model_func, current_solution
                )
                mgr_reward = self.manager.calculate_reward(
                    state, action, next_state, info, perf_reward, rel_reward
                )
            
            manager_rewards.append(mgr_reward)
            performance_rewards.append(perf_reward)
            reliability_rewards.append(rel_reward)
            
            total_reward += mgr_reward
            step_count += 1
            
            state = next_state
            
            if terminated or truncated:
                break
            
            # Update progress (less frequently)
            if progress_callback and step_count % 5 == 0:
                progress_callback(episode_num, step_count, total_reward)
        
        # Update agents with their experiences (batch update)
        if len(episode_states) > 0:
            self.manager.update(episode_states, episode_actions, manager_rewards)
            self.performance_agent.update(episode_states, episode_actions, performance_rewards)
            self.reliability_agent.update(episode_states, episode_actions, reliability_rewards)
        
        # Decay exploration (less frequently)
        if episode_num % 5 == 0:
            self.manager.decay_epsilon()
            self.performance_agent.decay_epsilon()
            self.reliability_agent.decay_epsilon()
        
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(step_count)
        
        return total_reward, step_count
    
    def _fast_performance_reward(self, state, action):
        """Fast performance reward without damage model"""
        # Use state features directly
        power_left = state[4:7]  # Power left for each PU
        power_reduced = state[7:10]  # Cumulative power reduced
        selected_pu_power = power_left[action]
        avg_power_reduced = np.mean(power_reduced)
        return selected_pu_power * 5.0 - avg_power_reduced * 10.0
    
    def _fast_reliability_reward(self, state, action):
        """Fast reliability reward without damage model - STRONG constraints"""
        rul = state[1:4]  # RUL for each PU (normalized)
        selected_pu_rul = rul[action]
        min_rul = np.min(rul)
        
        reward = selected_pu_rul * 30.0  # Increased reward for high RUL
        
        # CRITICAL: Prevent failures
        if min_rul < 0:
            reward -= 1000.0  # EXTREME penalty for failure
        elif min_rul < 0.05:  # Less than 5% RUL (critical)
            reward -= 500.0  # Very severe penalty
        elif min_rul < 0.1:  # Less than 10% RUL (warning)
            reward -= 200.0  # Severe penalty
        
        # Prevent using PU with low RUL
        if selected_pu_rul < 0.05:  # Less than 5% RUL
            reward -= 500.0  # Very severe penalty
        elif selected_pu_rul < 0.1:  # Less than 10% RUL
            reward -= 200.0  # Severe penalty
        elif selected_pu_rul < 0.2:  # Less than 20% RUL
            reward -= 50.0  # Moderate penalty
        
        # Reward for balanced RUL (prevent overusing one PU)
        rul_std = np.std(rul)
        reward -= rul_std * 20.0  # Stronger penalty for imbalance
        
        # Bonus if all PUs are above safe threshold
        if np.all(rul > 0.2):  # All above 20%
            reward += 50.0
        
        return reward
    
    def _fast_manager_reward(self, state, action, perf_reward, rel_reward):
        """Fast manager reward without damage model"""
        reward = 0.0
        # Check Fresh PU constraints
        fresh_pu_constraints = state[16:19]
        if np.any(fresh_pu_constraints > 0.5):
            required_pu = np.argmax(fresh_pu_constraints)
            if action == required_pu:
                reward += 10.0
            else:
                reward -= 20.0
        # Coordinate with co-workers
        reward += self.manager.performance_weight * perf_reward + self.manager.reliability_weight * rel_reward
        # Check available actions
        available_mask = state[13:16]
        if available_mask[action] < 0.5:
            reward -= 50.0
        return reward
    
    def train(self, progress_callback=None, fast_mode=True):
        """Train all agents - optimized for speed"""
        for episode in range(self.num_episodes):
            reward, length = self.train_episode(episode, progress_callback, fast_mode=fast_mode)
            # Less frequent printing for speed
            if (episode + 1) % 20 == 0 or episode == 0:
                if len(self.episode_rewards) > 0:
                    recent_rewards = self.episode_rewards[-min(10, len(self.episode_rewards)):]
                    avg_reward = np.mean(recent_rewards)
                    print(f"Episode {episode+1}/{self.num_episodes}, Avg Reward: {avg_reward:.2f}")
    
    def get_best_solution(self, use_eval_mode=True):
        """Get the best solution using trained policy with bias-weighted action selection"""
        # Set agents to eval mode for deterministic decisions
        if use_eval_mode:
            self.manager.set_eval_mode()
            self.performance_agent.set_eval_mode()
            self.reliability_agent.set_eval_mode()
        
        # Run one episode with greedy policy (no exploration)
        state, info = self.env.reset()
        
        solution_states = []
        solution_actions = []
        
        while True:
            available_actions = self.get_available_actions(state)
            
            if not available_actions:
                break
            
            # Get recommendations from all agents
            manager_probs = self.manager.policy_net(torch.FloatTensor(state).unsqueeze(0))
            perf_probs = self.performance_agent.policy_net(torch.FloatTensor(state).unsqueeze(0))
            rel_probs = self.reliability_agent.policy_net(torch.FloatTensor(state).unsqueeze(0))
            
            # Mask unavailable actions
            mask = torch.zeros(3)
            for action in available_actions:
                mask[action] = 1.0
            
            manager_probs = manager_probs * mask
            perf_probs = perf_probs * mask
            rel_probs = rel_probs * mask
            
            # Normalize
            manager_probs = manager_probs / (manager_probs.sum() + 1e-8)
            perf_probs = perf_probs / (perf_probs.sum() + 1e-8)
            rel_probs = rel_probs / (rel_probs.sum() + 1e-8)
            
            # Weighted combination based on bias
            # Manager gets base weight, co-workers get weights from manager's settings
            base_weight = 0.3  # Reduced manager weight
            perf_weight = self.manager.performance_weight
            rel_weight = self.manager.reliability_weight
            
            # RELIABILITY PRIORITY: If any PU has low RUL, STRONGLY increase reliability weight
            rul = state[1:4]
            min_rul = np.min(rul)
            if min_rul < 0.05:  # Any PU below 5% RUL (critical)
                rel_weight = rel_weight * 5.0  # 5x reliability weight
                perf_weight = perf_weight * 0.2  # Greatly reduce performance weight
                base_weight = base_weight * 0.5  # Reduce manager weight
            elif min_rul < 0.1:  # Any PU below 10% RUL (warning)
                rel_weight = rel_weight * 4.0  # 4x reliability weight
                perf_weight = perf_weight * 0.3  # Reduce performance weight
            elif min_rul < 0.2:  # Any PU below 20% RUL (caution)
                rel_weight = rel_weight * 2.5  # 2.5x reliability weight
                perf_weight = perf_weight * 0.6  # Slightly reduce performance weight
            
            # Normalize weights
            total_weight = base_weight + perf_weight + rel_weight
            base_weight = base_weight / total_weight
            perf_weight = perf_weight / total_weight
            rel_weight = rel_weight / total_weight
            
            # Combine probabilities
            combined_probs = (base_weight * manager_probs + 
                            perf_weight * perf_probs + 
                            rel_weight * rel_probs)
            
            # CRITICAL: Apply reliability mask - zero out probabilities for unsafe PUs
            reliability_mask = torch.ones(3)
            for a in range(3):
                if a in available_actions:
                    # Check if this PU has safe RUL
                    if rul[a] < 0.05:  # Less than 5% RUL
                        combined_probs[0, a] = combined_probs[0, a] * 0.01  # Almost zero
                    elif rul[a] < 0.1:  # Less than 10% RUL
                        combined_probs[0, a] = combined_probs[0, a] * 0.1  # Very low
                    elif rul[a] < 0.2:  # Less than 20% RUL
                        combined_probs[0, a] = combined_probs[0, a] * 0.5  # Reduced
            
            # Renormalize after masking
            combined_probs = combined_probs / (combined_probs.sum() + 1e-8)
            
            # Select action from combined distribution
            action = torch.multinomial(combined_probs, 1).item()
            
            # Ensure action is available and safe
            if action not in available_actions:
                # Fallback to safest available action
                if available_actions:
                    action = max(available_actions, key=lambda a: rul[a])
                else:
                    action = available_actions[0] if available_actions else 0
            
            solution_states.append(state)
            solution_actions.append(action)
            
            next_state, _, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                break
            
            state = next_state
        
        return self.env.get_final_solution()
    
    def save_models(self, base_path="models/rl_agents"):
        """Save all agent models"""
        os.makedirs(base_path, exist_ok=True)
        self.manager.save_model(f"{base_path}/manager.pth")
        self.performance_agent.save_model(f"{base_path}/performance.pth")
        self.reliability_agent.save_model(f"{base_path}/reliability.pth")
    
    def load_models(self, base_path="models/rl_agents"):
        """Load all agent models"""
        manager_loaded = self.manager.load_model(f"{base_path}/manager.pth")
        perf_loaded = self.performance_agent.load_model(f"{base_path}/performance.pth")
        rel_loaded = self.reliability_agent.load_model(f"{base_path}/reliability.pth")
        return manager_loaded and perf_loaded and rel_loaded
    
    def quick_inference(self, progress_bar=None):
        """Quick inference mode - uses pre-trained models without training"""
        if progress_bar:
            progress_bar.progress(0.1, text='Loading pre-trained RL models...')
        
        # Load models if available
        models_loaded = self.load_models()
        
        if not models_loaded:
            if progress_bar:
                progress_bar.progress(0.5, text='No pre-trained models found. Quick training...')
            # If no pre-trained models, do a quick training (few episodes, fast mode)
            self.num_episodes = min(5, self.num_episodes)  # Very quick training
            self.train(progress_callback=None, fast_mode=True)
        else:
            if progress_bar:
                progress_bar.progress(0.3, text='Using pre-trained models for inference...')
        
        if progress_bar:
            progress_bar.progress(0.8, text='Generating optimal PU allocation...')
        
        # Get solution using trained models
        solution = self.get_best_solution(use_eval_mode=True)
        
        if progress_bar:
            progress_bar.progress(1.0, text='PU allocation complete!')
        
        return solution

