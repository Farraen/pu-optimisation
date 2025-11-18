"""
RL Environment for PU Selection Problem
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd


class PUSelectionEnv(gym.Env):
    """
    Custom Gymnasium environment for Power Unit (PU) selection optimization.
    Each episode represents a full season of races.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, track_data, damage_model_func, max_pu_usage=3, render_mode=None):
        super().__init__()
        
        self.original_track_data = track_data.copy()
        self.track_data = track_data.copy()
        self.damage_model_func = damage_model_func
        self.max_pu_usage = max_pu_usage
        self.render_mode = render_mode
        
        # Get races to optimize (those without PU Actual)
        self.races_to_optimize = track_data[track_data["PU Actual"].isna()].index.tolist()
        self.num_races = len(self.races_to_optimize)
        
        if self.num_races == 0:
            self.num_races = len(track_data)
            self.races_to_optimize = list(range(len(track_data)))
        
        # Action space: discrete choice of PU (1, 2, or 3) for each race
        self.action_space = spaces.Discrete(3)  # 0->PU1, 1->PU2, 2->PU3
        
        # State space: 
        # - Current race index (normalized)
        # - RUL for each PU (3 values, normalized)
        # - Power left for each PU (3 values, normalized)
        # - Cumulative power reduced for each PU (3 values, normalized)
        # - Race characteristics: Distance, MinTemp, MaxTemp (normalized)
        # - Available PUs mask (3 binary values)
        # - Fresh PU constraints (3 binary values)
        # Total: 1 + 3 + 3 + 3 + 3 + 3 + 3 = 19 features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )
        
        self.reset()
    
    def _get_available_pus(self, race_idx):
        """Get list of available PUs for a given race"""
        df_temp = self.track_data.copy()
        pu_available = [1, 2, 3]
        
        # Remove failed PUs
        if race_idx > 0:
            PU_failed = df_temp.loc[:race_idx-1, "PU Failures"].dropna().unique().tolist()
            for pu in PU_failed:
                if int(pu) in pu_available:
                    pu_available.remove(int(pu))
        
        # Remove PUs assigned to future races (Fresh PU constraint)
        if "Fresh PU" in df_temp.columns:
            future_races = df_temp.loc[race_idx+1:, "Fresh PU"].dropna().unique().tolist()
            for pu in future_races:
                if int(pu) in pu_available:
                    pu_available.remove(int(pu))
        
        return pu_available
    
    def _get_state(self):
        """Extract current state observation"""
        if self.current_race_idx >= self.num_races:
            # Episode finished, return terminal state
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        race_idx = self.races_to_optimize[self.current_race_idx]
        df_temp = self.track_data.copy()
        
        # Get current PU states
        current_solution = df_temp["PU Projection"].to_numpy()
        
        # Initialize default values
        rul_pu1, rul_pu2, rul_pu3 = 1.0, 1.0, 1.0
        power_pu1, power_pu2, power_pu3 = 1.0, 1.0, 1.0
        power_red_pu1, power_red_pu2, power_red_pu3 = 0.0, 0.0, 0.0
        
        # Only calculate if we have valid solution
        if not np.isnan(current_solution).all():
            try:
                _, _, PowerLeft, RUL, PowerReduced = self.damage_model_func(current_solution)
                
                # Get latest RUL for each PU
                pu1_races = np.where(current_solution == 1)[0]
                pu2_races = np.where(current_solution == 2)[0]
                pu3_races = np.where(current_solution == 3)[0]
                
                if len(pu1_races) > 0:
                    last_idx = pu1_races[-1]
                    rul_vals = RUL[RUL["Index"] == last_idx]["RUL"].values
                    rul_pu1 = (rul_vals[0] / 100.0) if len(rul_vals) > 0 else 1.0
                    power_vals = PowerLeft[PowerLeft["Index"] == last_idx]["PowerLeft"].values
                    power_pu1 = (power_vals[0] / 450.0) if len(power_vals) > 0 else 1.0
                    power_red_vals = PowerReduced[PowerReduced["Index"] == last_idx]["PowerReduced"].values
                    power_red_pu1 = (power_red_vals[0] / 50.0) if len(power_red_vals) > 0 else 0.0
                
                if len(pu2_races) > 0:
                    last_idx = pu2_races[-1]
                    rul_vals = RUL[RUL["Index"] == last_idx]["RUL"].values
                    rul_pu2 = (rul_vals[0] / 100.0) if len(rul_vals) > 0 else 1.0
                    power_vals = PowerLeft[PowerLeft["Index"] == last_idx]["PowerLeft"].values
                    power_pu2 = (power_vals[0] / 450.0) if len(power_vals) > 0 else 1.0
                    power_red_vals = PowerReduced[PowerReduced["Index"] == last_idx]["PowerReduced"].values
                    power_red_pu2 = (power_red_vals[0] / 50.0) if len(power_red_vals) > 0 else 0.0
                
                if len(pu3_races) > 0:
                    last_idx = pu3_races[-1]
                    rul_vals = RUL[RUL["Index"] == last_idx]["RUL"].values
                    rul_pu3 = (rul_vals[0] / 100.0) if len(rul_vals) > 0 else 1.0
                    power_vals = PowerLeft[PowerLeft["Index"] == last_idx]["PowerLeft"].values
                    power_pu3 = (power_vals[0] / 450.0) if len(power_vals) > 0 else 1.0
                    power_red_vals = PowerReduced[PowerReduced["Index"] == last_idx]["PowerReduced"].values
                    power_red_pu3 = (power_red_vals[0] / 50.0) if len(power_red_vals) > 0 else 0.0
            except:
                # If damage model fails, use default values
                pass
        
        # Race characteristics
        race_row = df_temp.iloc[race_idx]
        distance = race_row.get("Distance", 300) / 400.0  # Normalize
        min_temp = race_row.get("MinTemp", 20) / 50.0  # Normalize
        max_temp = race_row.get("MaxTemp", 30) / 50.0  # Normalize
        
        # Available PUs mask
        available_pus = self._get_available_pus(race_idx)
        pu1_available = 1.0 if 1 in available_pus else 0.0
        pu2_available = 1.0 if 2 in available_pus else 0.0
        pu3_available = 1.0 if 3 in available_pus else 0.0
        
        # Fresh PU constraints
        fresh_pu = race_row.get("Fresh PU", np.nan)
        fresh_pu1 = 1.0 if not np.isnan(fresh_pu) and fresh_pu == 1 else 0.0
        fresh_pu2 = 1.0 if not np.isnan(fresh_pu) and fresh_pu == 2 else 0.0
        fresh_pu3 = 1.0 if not np.isnan(fresh_pu) and fresh_pu == 3 else 0.0
        
        # Current race index (normalized)
        race_progress = self.current_race_idx / max(self.num_races, 1)
        
        state = np.array([
            race_progress,
            rul_pu1, rul_pu2, rul_pu3,
            power_pu1, power_pu2, power_pu3,
            power_red_pu1, power_red_pu2, power_red_pu3,
            distance, min_temp, max_temp,
            pu1_available, pu2_available, pu3_available,
            fresh_pu1, fresh_pu2, fresh_pu3
        ], dtype=np.float32)
        
        return state
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset to initial state
        self.current_race_idx = 0
        # Create a fresh copy of track data to avoid modifying original
        self.track_data = self.original_track_data.copy()
        
        # Initialize PU Projection if needed
        if "PU Projection" not in self.track_data.columns or self.track_data["PU Projection"].isna().all():
            self.track_data["PU Projection"] = np.nan
        
        # Fill in actual values
        if not self.track_data["PU Actual"].isna().all():
            self.track_data.loc[~self.track_data["PU Actual"].isna(), "PU Projection"] = \
                self.track_data.loc[~self.track_data["PU Actual"].isna(), "PU Actual"]
        
        # Apply Fresh PU constraints
        if "Fresh PU" in self.track_data.columns:
            fresh_mask = ~self.track_data["Fresh PU"].isna()
            self.track_data.loc[fresh_mask, "PU Projection"] = self.track_data.loc[fresh_mask, "Fresh PU"]
        
        observation = self._get_state()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.current_race_idx >= self.num_races:
            # Episode already finished
            return self._get_state(), 0.0, True, False, {}
        
        race_idx = self.races_to_optimize[self.current_race_idx]
        
        # Convert action (0,1,2) to PU number (1,2,3)
        pu_selected = action + 1
        
        # Check if PU is available
        available_pus = self._get_available_pus(race_idx)
        if pu_selected not in available_pus:
            # Invalid action - use first available PU
            if available_pus:
                pu_selected = available_pus[0]
            else:
                pu_selected = 1  # Fallback
        
        # Apply the selection
        self.track_data.loc[race_idx, "PU Projection"] = pu_selected
        
        # Move to next race
        self.current_race_idx += 1
        
        # Calculate rewards (will be computed by hierarchical agents)
        reward = 0.0  # Base reward, will be modified by agents
        
        # Check if episode is done
        terminated = self.current_race_idx >= self.num_races
        truncated = False
        
        observation = self._get_state()
        info = {
            "race_idx": race_idx,
            "pu_selected": pu_selected,
            "current_race_idx": self.current_race_idx
        }
        
        return observation, reward, terminated, truncated, info
    
    def get_final_solution(self):
        """Get the final PU allocation solution"""
        return self.track_data["PU Projection"].to_numpy()
    
    def render(self):
        pass

