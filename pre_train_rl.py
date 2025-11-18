"""
Pre-training script for hierarchical RL agents
Trains on diverse scenarios to create generalizable models
"""
import numpy as np
import pandas as pd
import os
from rl_environment import PUSelectionEnv
from rl_agents import HierarchicalRLCoordinator
import random

def create_diverse_scenarios(base_track_data, num_scenarios=50):
    """Create diverse training scenarios with different constraints"""
    scenarios = []
    
    for i in range(num_scenarios):
        scenario = base_track_data.copy()
        
        # Randomly set some PU Actual values (simulating completed races)
        num_completed = random.randint(0, len(scenario) // 3)
        if num_completed > 0:
            completed_indices = random.sample(range(len(scenario)), num_completed)
            for idx in completed_indices:
                scenario.loc[idx, "PU Actual"] = random.choice([1, 2, 3])
        
        # Randomly set some Fresh PU assignments
        num_fresh = random.randint(0, min(3, len(scenario) // 4))
        if num_fresh > 0:
            fresh_indices = random.sample(range(len(scenario)), num_fresh)
            for idx in fresh_indices:
                scenario.loc[idx, "Fresh PU"] = random.choice([1, 2, 3])
        
        # Randomly set some PU failures
        if random.random() < 0.3:  # 30% chance of failures
            num_failures = random.randint(1, 2)
            failure_indices = random.sample(range(len(scenario) - 2), num_failures)
            for idx in failure_indices:
                scenario.loc[idx, "PU Failures"] = random.choice([1, 2, 3])
        
        scenarios.append(scenario)
    
    return scenarios

def pre_train_agents(track_data, damage_model_func, num_scenarios=15, episodes_per_scenario=10, retrain=False):
    """Pre-train RL agents on diverse scenarios - optimized for speed
    
    Args:
        track_data: Base track data
        damage_model_func: Damage model function
        num_scenarios: Number of diverse scenarios to train on
        episodes_per_scenario: Episodes per scenario
        retrain: If True, load existing models and continue training (fine-tuning)
    """
    print("Creating diverse training scenarios...")
    scenarios = create_diverse_scenarios(track_data, num_scenarios)
    
    # Create initial environment
    env = PUSelectionEnv(
        track_data=scenarios[0],
        damage_model_func=damage_model_func,
        max_pu_usage=3
    )
    
    # Create coordinator
    coordinator = HierarchicalRLCoordinator(
        env=env,
        damage_model_func=damage_model_func,
        num_episodes=episodes_per_scenario,
        use_pretrained=retrain  # Load existing models if retraining
    )
    
    if retrain:
        print(f"Retraining on {num_scenarios} diverse scenarios (continuing from existing models)...")
    else:
        print(f"Training on {num_scenarios} diverse scenarios...")
    total_episodes = 0
    
    for scenario_idx, scenario in enumerate(scenarios):
        # Update environment with new scenario
        coordinator.env.original_track_data = scenario.copy()
        coordinator.env.track_data = scenario.copy()
        coordinator.env.races_to_optimize = scenario[scenario["PU Actual"].isna()].index.tolist()
        coordinator.env.num_races = len(coordinator.env.races_to_optimize)
        
        if coordinator.env.num_races == 0:
            continue
        
        # Train on this scenario (use fast mode)
        coordinator.train(progress_callback=None, fast_mode=True)
        total_episodes += episodes_per_scenario
        
        if (scenario_idx + 1) % 5 == 0 or scenario_idx == 0:  # More frequent updates
            if len(coordinator.episode_rewards) >= episodes_per_scenario:
                avg_reward = np.mean(coordinator.episode_rewards[-episodes_per_scenario:])
                print(f"Completed {scenario_idx + 1}/{num_scenarios} scenarios. "
                      f"Recent avg reward: {avg_reward:.2f}")
    
    action = "Retraining" if retrain else "Pre-training"
    print(f"\n{action} complete! Total episodes: {total_episodes}")
    print("Saving models...")
    
    # Save trained models
    coordinator.save_models()
    print("Models saved to models/rl_agents/")
    
    return coordinator

if __name__ == "__main__":
    # This would be called separately to pre-train models
    # For now, it's a utility script
    print("Pre-training script for RL agents")
    print("Run this separately to create pre-trained models")

