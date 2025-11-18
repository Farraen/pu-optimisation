# Developed by Farraen
# Date 2018
# Migrated to python 2023

import os, sys
import time
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*use_container_width.*')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata
from PIL import Image
from openai import OpenAI
import json
import re
import ast
import statistics
from rl_environment import PUSelectionEnv
from rl_agents import HierarchicalRLCoordinator
import os


# ---------  System cache  -----------------

if 'pu_fitness_trace' not in st.session_state:
    st.session_state.pu_fitness_trace = []

if 'pu_results' not in st.session_state:
    st.session_state.pu_results = {}
elif not isinstance(st.session_state.pu_results, dict):
    # Convert old list format to dict format
    st.session_state.pu_results = {}

if 'df_1' not in st.session_state:
    st.session_state.df_1 = []

if 'pu_bias' not in st.session_state:
    st.session_state.pu_bias = 2

if 'num_episodes' not in st.session_state:
    st.session_state.num_episodes = 50

if "pu_iter" not in st.session_state:
    st.session_state.pu_iter = []

if 'rl_coordinator' not in st.session_state:
    st.session_state.rl_coordinator = None

if 'bias_changed' not in st.session_state:
    st.session_state.bias_changed = False

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Unified chat messages with agent tags
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Current prompt
if "chat_prompt" not in st.session_state:
    st.session_state.chat_prompt = []

# RL Agent Settings - Manager (Chief Engineer can modify)
if "rl_manager_settings" not in st.session_state:
    st.session_state.rl_manager_settings = {
        "performance_weight": 0.3,
        "reliability_weight": 0.3,
        "max_pu_usage": 3,
        "learning_rate": 0.003,
        "gamma": 0.99,
        "fresh_pu_reward": 10.0,
        "fresh_pu_penalty": -20.0,
        "low_rul_penalty_5pct": -300.0,
        "low_rul_penalty_10pct": -100.0
    }

# RL Agent Settings - Performance (Performance agent can modify)
if "rl_performance_settings" not in st.session_state:
    st.session_state.rl_performance_settings = {
        "learning_rate": 0.003,
        "gamma": 0.99,
        "power_reward_scale": 5.0,
        "power_reduced_penalty_scale": 10.0,
        "power_loss_normalization": 100.0,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995
    }

# RL Agent Settings - Reliability (Reliability agent can modify)
if "rl_reliability_settings" not in st.session_state:
    st.session_state.rl_reliability_settings = {
        "learning_rate": 0.003,
        "gamma": 0.99,
        "rul_reward_scale": 30.0,
        "rul_imbalance_penalty": 15.0,
        "rul_threshold_critical": 0.05,  # 5%
        "rul_threshold_warning": 0.10,   # 10%
        "rul_threshold_caution": 0.20,   # 20%
        "failure_penalty": -2000.0,
        "critical_rul_penalty": -800.0,
        "warning_rul_penalty": -400.0,
        "caution_rul_penalty": -100.0,
        "safe_operation_bonus": 100.0,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995
    }


# --------  For page layout  ---------------
st.set_page_config(layout="wide",initial_sidebar_state="collapsed")




st.markdown("""
<style>
.title_medium {
    font-size:20px !important;
}
 .text_small {
    font-size:12px !important;
}           
</style>
""", unsafe_allow_html=True)

def st_title(text):
    st.markdown(f'<p class="title_medium">{text}</p>', unsafe_allow_html=True)

def st_text(text):
    st.markdown(f'<p class="text_small">{text}</p>', unsafe_allow_html=True)

@st.cache_resource
def read_image(img_path):
    im = Image.open(img_path)
    image = np.array(im)
    return image

# ---------- Open AI ----------------------

# OpenAI 
@st.cache_resource 
def connect_openAI():
    # Use Streamlit secrets or environment variable for API key
    # For local development, set in .streamlit/secrets.toml
    # For production, set in Streamlit Cloud secrets
    openai_api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in .streamlit/secrets.toml or environment variables.")
        st.stop()
    openai_client = OpenAI(api_key=openai_api_key)
    return openai_client

client = connect_openAI()


# --------- Common functions --------------

def interp2(X,Y,Z,Xv,Yv):
    length_values = len(X) * len(Y)
    x_grid, y_grid = np.meshgrid(X, Y)   
    points = np.empty((length_values, 2))
    values = Z.flatten()
    points[:, 0] = x_grid.flatten()
    points[:, 1] = y_grid.flatten()
    grid_z1 = griddata(points, values, (Xv,Yv), method='cubic')

    return grid_z1

def DamageModel(x):

    dtt = st.session_state.df_1

    ICE_RUL  = [80, 70, 95]
    ICE_KW  = [425, 400, 450]

    # Piston clearance/km temp
    DamageModel_X = [0.1, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 41.0]
    DamageModel_Z = [2.0, 3.0, 4.0, 6.0, 9.0, 14.0, 19.0, 25.0, 32.0]
    DamageModel_Z = [item * 0.000001 for item in DamageModel_Z]

    # Power degradation due to piston clearance
    PerfModel_X = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.015]
    PerfModel_Z = [0.1,   1,     2,     3,     5,     7,     8,     10,    15,    30]

    # RUL loss based on ICE power loss and mileage per race
    RULModel_X = np.array([150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
    RULModel_Y = np.array([0.0, 2.0, 5.0, 10.0, 20.0, 25.0])
    RULModel_Z = np.array([[7.5, 7.0, 9.5, 11.5, 19.0, 32.0],
                  [6.5, 7.5, 9.0, 11.5, 15.5, 24.5],
                  [6.0, 6.5, 7.0, 10.0, 14.0, 17.0],
                  [4.5, 6.5, 7.0, 10.0, 11.5, 13.0],
                  [3.5, 5.0, 6.5, 8.50, 9.00, 10.0],
                  [1.5, 3.5, 5.5, 7.00, 8.00, 7.50]])
    RULModel_XX, RULModel_YY = np.meshgrid(RULModel_X, RULModel_Y) 

    # Average temperature in the race month
    AmbTemp = dtt["MinTemp"] + dtt["MaxTemp"]/2
    
    # Calculate ICE damage for each race
    interp_func = interp1d(DamageModel_X, DamageModel_Z)
    DamagePerKM = interp_func(AmbTemp)
    dt_DamagePerKM = pd.DataFrame(DamagePerKM,columns=["DamagePerKM"])
    dtt['DamageThisRace'] = dtt['Distance'] * dt_DamagePerKM['DamagePerKM']
    
    # Calculate the ICE power reduction
    interp_func2 = interp1d(PerfModel_X, PerfModel_Z)
    PowerReducedThisRace = interp_func2(dtt['DamageThisRace'])

    # Get individual power reduction for each ICE
    PowerRedICE1 = PowerReducedThisRace[np.where(x == 1)[0]]
    PowerRedICE2 = PowerReducedThisRace[np.where(x == 2)[0]]
    PowerRedICE3 = PowerReducedThisRace[np.where(x == 3)[0]]

    # Get distance ran for each ICE
    DistanceICE1 = dtt.loc[np.where(x == 1)[0],'Distance'].to_numpy()
    DistanceICE2 = dtt.loc[np.where(x == 2)[0],'Distance'].to_numpy()
    DistanceICE3 = dtt.loc[np.where(x == 3)[0],'Distance'].to_numpy()

    # RUL reduction for each ICE based on damage model
    RULReducedICE1 = interp2(RULModel_X,RULModel_Y,RULModel_Z,DistanceICE1,PowerRedICE1)
    RULReducedICE2 = interp2(RULModel_X,RULModel_Y,RULModel_Z,DistanceICE2,PowerRedICE2)
    RULReducedICE3 = interp2(RULModel_X,RULModel_Y,RULModel_Z,DistanceICE3,PowerRedICE3)

    # Calculate cummulative RUL reduction
    RULARR1 = pd.DataFrame({"Index":np.where(x == 1)[0],"CumSum":np.cumsum(RULReducedICE1)})
    RULARR2 = pd.DataFrame({"Index":np.where(x == 2)[0],"CumSum":np.cumsum(RULReducedICE2)})
    RULARR3 = pd.DataFrame({"Index":np.where(x == 3)[0],"CumSum":np.cumsum(RULReducedICE3)})
    RULARR = pd.concat([RULARR1, RULARR2, RULARR3], ignore_index=True)
    RUL_Reduced = RULARR.sort_values('Index')
    RUL_Reduced.reset_index(drop=True, inplace=True)

    # Calculate cummulative Power reduction
    RULARR1 = pd.DataFrame({"Index":np.where(x == 1)[0],"PowerReduced":np.cumsum(PowerRedICE1)})
    RULARR2 = pd.DataFrame({"Index":np.where(x == 2)[0],"PowerReduced":np.cumsum(PowerRedICE2)})
    RULARR3 = pd.DataFrame({"Index":np.where(x == 3)[0],"PowerReduced":np.cumsum(PowerRedICE3)})
    RULARR = pd.concat([RULARR1, RULARR2, RULARR3], ignore_index=True)
    PowerReduced = RULARR.sort_values('Index')
    PowerReduced.reset_index(drop=True, inplace=True)

    # Calculate RUL left for each ICE and store in data table
    RULARR1 = pd.DataFrame({"Index":np.where(x == 1)[0],"RUL":ICE_RUL[0]-np.cumsum(RULReducedICE1)})
    RULARR2 = pd.DataFrame({"Index":np.where(x == 2)[0],"RUL":ICE_RUL[1]-np.cumsum(RULReducedICE2)})
    RULARR3 = pd.DataFrame({"Index":np.where(x == 3)[0],"RUL":ICE_RUL[2]-np.cumsum(RULReducedICE3)})
    RULARR = pd.concat([RULARR1, RULARR2, RULARR3], ignore_index=True)
    RUL = RULARR.sort_values('Index')
    RUL.reset_index(drop=True, inplace=True)

    # Calculate Power left for each ICE and store in data table
    RULARR1 = pd.DataFrame({"Index":np.where(x == 1)[0],"PowerLeft":ICE_KW[0]-np.cumsum(PowerRedICE1)})
    RULARR2 = pd.DataFrame({"Index":np.where(x == 2)[0],"PowerLeft":ICE_KW[1]-np.cumsum(PowerRedICE2)})
    RULARR3 = pd.DataFrame({"Index":np.where(x == 3)[0],"PowerLeft":ICE_KW[2]-np.cumsum(PowerRedICE3)})
    RULARR = pd.concat([RULARR1, RULARR2, RULARR3], ignore_index=True)
    PowerLeft = RULARR.sort_values('Index')
    PowerLeft.reset_index(drop=True, inplace=True)

    # Calculate fitness function (inverse of penalty function)
    PowerLoss = -(np.sum(PowerRedICE1) + np.sum(PowerRedICE2) + np.sum(PowerRedICE3))

    failedPU = RUL.loc[np.where(RUL["RUL"] < 0)[0],'Index'].to_numpy()

    pu_bias = (st.session_state.pu_bias-1)/2
    fitness_value = PowerLoss - (pu_bias)*np.sum(failedPU)


    return fitness_value, PowerLoss, PowerLeft, RUL, PowerReduced
        
def plot_results():

    # Load results
    if isinstance(st.session_state.pu_results, dict) and len(st.session_state.pu_results) > 0:

        fig = go.Figure()
        fig2 = go.Figure()
        
        # Get all run keys (run_0, run_1, etc.) and best key
        run_keys = sorted([k for k in st.session_state.pu_results.keys() if isinstance(k, str) and k.startswith('run_')])
        has_best = 'best' in st.session_state.pu_results
        
        # Get numeric keys for backward compatibility (old format)
        numeric_keys = sorted([k for k in st.session_state.pu_results.keys() if isinstance(k, (int, float))])
        
        # Plot 1: RUL over races
        # First, plot all run results as grey lines
        for run_key in run_keys:
            if run_key in st.session_state.pu_results:
                dts = st.session_state.pu_results[run_key]
                fig.add_trace(go.Scatter(
                    y=dts['RUL'],
                    mode='lines',
                    name='Other runs',
                    showlegend=False,  # Don't show legend for individual runs
                    opacity=0.5,
                    line=dict(
                        width=1,
                        color='grey')
                ))
        
        # Plot numeric keys as grey (backward compatibility)
        for key in numeric_keys:
            if key in st.session_state.pu_results:
                dts = st.session_state.pu_results[key]
                fig.add_trace(go.Scatter(
                    y=dts['RUL'],
                    mode='lines',
                    name='Previous solutions',
                    showlegend=False,
                    opacity=0.5,
                    line=dict(
                        width=1,
                        color='grey')
                ))
        
        # Plot best solution last (so it appears on top)
        if has_best:
            df_best = st.session_state.pu_results['best']
            fig.add_trace(go.Scatter(
                y=df_best['RUL'],
                mode='lines',
                name='Best PU allocation',
                line=dict(
                    width=4,
                    color='blue')
            ))
        elif len(numeric_keys) > 0:
            # Fallback to last numeric key if no 'best' key
            key = numeric_keys[-1]
            df_best = st.session_state.pu_results[key]
            fig.add_trace(go.Scatter(
                y=df_best['RUL'],
                mode='lines',
                name='Best PU allocation',
                line=dict(
                    width=4,
                    color='blue')
            ))
        
        fig.add_hline(y=0, line_width=3, line_dash="dash", line_color="red", annotation_text="RUL threshold",annotation_position="bottom left")
        fig.update_yaxes(title_text='PU RUL (Remaining Useful Life)')
        fig.update_xaxes(title_text='Race List for the season',tickmode='linear')
        fig.update_layout(legend=dict(
            orientation="v",
            yanchor="auto",
            y=1,
            xanchor="right",
            x=1
        ))
        fig.update_layout(height=300,margin=dict(l=20, r=20, t=20, b=20))
        st.session_state.iterCompare_placeholder.plotly_chart(fig,use_container_width=True,height=300)

        # Plot 2: Power reduction per PU
        # First, plot all run results as grey lines
        for run_key in run_keys:
            if run_key in st.session_state.pu_results:
                dts = st.session_state.pu_results[run_key]
                
                # Get list of PUs
                pu_list = dts["PU Projection"].unique().tolist()
                pu_list = [p for p in pu_list if not pd.isna(p)]
                pu_list.sort()

                if len(pu_list) == 0:
                    continue

                # Find all max reduced for each PU
                maxpowerreduced = []
                for k in pu_list:
                    pu_indices = np.where(dts["PU Projection"] == k)[0]
                    if len(pu_indices) > 0:
                        mx = np.max(dts.loc[pu_indices,'PowerReduced'].to_numpy())
                        maxpowerreduced.append(mx)

                if len(maxpowerreduced) > 0:
                    fig2.add_trace(go.Scatter(
                        x=pu_list,
                        y=maxpowerreduced,
                        mode='lines',
                        name='Other runs',
                        showlegend=False,
                        opacity=0.5,
                        line=dict(
                            width=1,
                            color='grey')
                    ))
        
        # Plot numeric keys as grey (backward compatibility)
        for key in numeric_keys:
            if key in st.session_state.pu_results:
                dts = st.session_state.pu_results[key]
                
                pu_list = dts["PU Projection"].unique().tolist()
                pu_list = [p for p in pu_list if not pd.isna(p)]
                pu_list.sort()

                if len(pu_list) == 0:
                    continue

                maxpowerreduced = []
                for k in pu_list:
                    pu_indices = np.where(dts["PU Projection"] == k)[0]
                    if len(pu_indices) > 0:
                        mx = np.max(dts.loc[pu_indices,'PowerReduced'].to_numpy())
                        maxpowerreduced.append(mx)

                if len(maxpowerreduced) > 0:
                    fig2.add_trace(go.Scatter(
                        x=pu_list,
                        y=maxpowerreduced,
                        mode='lines',
                        name='Previous solutions',
                        showlegend=False,
                        opacity=0.5,
                        line=dict(
                            width=1,
                            color='grey')
                    ))
        
        # Plot best solution last (so it appears on top)
        if has_best:
            df_best = st.session_state.pu_results['best']
        elif len(numeric_keys) > 0:
            df_best = st.session_state.pu_results[numeric_keys[-1]]
        else:
            df_best = None
        
        if df_best is not None:
            pu_list = df_best["PU Projection"].unique().tolist()
            pu_list = [p for p in pu_list if not pd.isna(p)]
            pu_list.sort()

            if len(pu_list) > 0:
                maxpowerreduced = []
                for k in pu_list:
                    pu_indices = np.where(df_best["PU Projection"] == k)[0]
                    if len(pu_indices) > 0:
                        mx = np.max(df_best.loc[pu_indices,'PowerReduced'].to_numpy())
                        maxpowerreduced.append(mx)

                if len(maxpowerreduced) > 0:
                    fig2.add_trace(go.Scatter(
                        x=pu_list,
                        y=maxpowerreduced,
                        mode='lines',
                        name='Best PU Allocation',
                        line=dict(
                            width=4,
                            color='blue')
                    ))
                    
                    ymax = max(maxpowerreduced)
                    ymin = min(maxpowerreduced)
                    fig2.add_hline(y=ymax, line_width=3, line_dash="dash", line_color="red", annotation_text="Max of best solution",annotation_position="bottom left")
                    fig2.add_hline(y=ymin, line_width=3, line_dash="dash", line_color="red", annotation_text="Min of best solution",annotation_position="bottom left")
        
        fig2.update_yaxes(title_text='Power Reduction End Season (kW)')
        fig2.update_xaxes(title_text='Power Unit',tickmode='linear')
        fig2.update_layout(legend=dict(
            orientation="v",
            yanchor="auto",
            y=1,
            xanchor="right",
            x=1))
        fig2.update_layout(height=300,margin=dict(l=20, r=20, t=20, b=20))
        st.session_state.PUCompare_placeholder.plotly_chart(fig2,use_container_width=True)

        #st_text('Plot shows the prediction of PU degradation for the race season using optimised PU allocation.')    
        #st_text('Plot shows the power loss due to degradation for all power units')

def plot_iter():
    if len(st.session_state.pu_fitness_trace) > 0:
        dta = pd.DataFrame(st.session_state.pu_fitness_trace,columns=["value"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=dta['value'],
            mode='markers+lines',
        ))
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        fig.update_yaxes(title_text='Episode Reward')
        fig.update_xaxes(title_text='Episode')
        fig.update_layout(height=300)

        st.session_state.pu_iter_placeholder.plotly_chart(fig,use_container_width=True,height=300)

def change_bias():
    pu_bias = st.session_state.slider
    if pu_bias == 'High Performance':
        st.session_state.pu_bias = 1
    elif pu_bias == 'Longer RUL':
        st.session_state.pu_bias = 10
    else:
        st.session_state.pu_bias = int(pu_bias)
    
    # Set flag to trigger re-optimization on next run
    st.session_state.bias_changed = True

# --------  For optimisation  -------------

def make_full_solution(solution):
    dff = st.session_state.df_1.copy()
    solutionFull = dff["PU Projection"].to_numpy().copy()
    
    # First, fill in PU Actual values (races that have been completed)
    if not dff["PU Actual"].isna().all():
        tracks_left_idx = dff["PU Actual"].isna()
        actual = dff["PU Actual"].to_numpy()
        solutionFull[~tracks_left_idx] = actual[~tracks_left_idx]
        
        # For remaining races, use the solution from RL
        remaining_indices = np.where(tracks_left_idx)[0]
        for i, idx in enumerate(remaining_indices):
            if i < len(solution):
                solutionFull[idx] = solution[i]
    else:
        # No actual values, use the RL solution
        solutionFull = solution
    
    # Then, override with Fresh PU values (user-specified PU assignments)
    if "Fresh PU" in dff.columns:
        assigned_pu = dff["Fresh PU"].to_numpy()
        for idx in range(len(assigned_pu)):
            if not np.isnan(assigned_pu[idx]):
                solutionFull[idx] = int(assigned_pu[idx])
    
    return solutionFull

def progress_callback(episode, step, reward, progress_bar=None):
    """Callback for RL training progress"""
    st.session_state.pu_iter = episode + 1
    progress = (episode + 1) / st.session_state.num_episodes
    
    if progress_bar:
        progress_bar.progress(progress, text=f'Training RL agents... Episode {episode+1}/{st.session_state.num_episodes}')
    
    # Update fitness trace
    if len(st.session_state.pu_fitness_trace) <= episode:
        st.session_state.pu_fitness_trace.append(reward)
    else:
        st.session_state.pu_fitness_trace[episode] = reward

def quick_optimisation(progress_bar=None):
    """Quick optimization using pre-trained models - runs 10 times and selects best"""
    
    # Check if pre-trained models exist
    models_exist = os.path.exists("models/rl_agents/manager.pth")
    
    if not models_exist:
        st.warning("⚠ No pre-trained models found. Please click 'Pre-train Models' button first for best results.")
        st.info("Running quick training (5 episodes) as fallback...")
    
    # Only optimise track with no actual PU results
    df_temp = st.session_state.df_1.copy()
    tracks_to_optimize_idx = df_temp[df_temp["PU Actual"].isna()].index.tolist()
    
    if len(tracks_to_optimize_idx) == 0:
        st.error("No races to optimize")
        return
    
    # Run 10 iterations to get multiple solutions
    num_runs = 10
    all_solutions = []
    
    if progress_bar:
        progress_bar.progress(0.0, text=f'Running RL optimization {num_runs} times...')
    
    try:
        for run_idx in range(num_runs):
            # Create RL environment (fresh copy for each run)
            env = PUSelectionEnv(
                track_data=df_temp.copy(),
                damage_model_func=DamageModel,
                max_pu_usage=3
            )
            
            # Create hierarchical RL coordinator with pre-trained models
            coordinator = HierarchicalRLCoordinator(
                env=env,
                damage_model_func=DamageModel,
                num_episodes=0,  # No training needed if models exist
                use_pretrained=True
            )
            
            # Apply RL settings from session state (can be modified via chat)
            manager_settings = st.session_state.get('rl_manager_settings', {})
            coordinator.manager.update_reward_weights(
                performance_weight=manager_settings.get('performance_weight', 0.3),
                reliability_weight=manager_settings.get('reliability_weight', 0.3)
            )
            
            # Note: Other settings like learning_rate, gamma would need to be applied
            # when creating the agents, but for now we focus on reward weights which
            # can be updated dynamically
            
            # Get solution from this run
            solution = coordinator.get_best_solution(use_eval_mode=True)
            full_solution = make_full_solution(solution)
            
            # Calculate metrics for this solution
            Fitness, PowerLoss, PowerLeft, RUL, PowerReduced = DamageModel(full_solution)
            
            # Store solution and metrics
            all_solutions.append({
                'solution': full_solution,
                'fitness': Fitness,
                'PowerLoss': PowerLoss,
                'PowerLeft': PowerLeft,
                'RUL': RUL,
                'PowerReduced': PowerReduced
            })
            
            if progress_bar:
                progress = (run_idx + 1) / num_runs
                progress_bar.progress(progress, text=f'Running RL optimization {run_idx + 1}/{num_runs}...')
        
        # Select best solution based on fitness (higher is better)
        best_idx = max(range(len(all_solutions)), key=lambda i: all_solutions[i]['fitness'])
        best_solution_data = all_solutions[best_idx]
        
        # Update main dataframe with best solution
        st.session_state.df_1["PU Projection"] = best_solution_data['solution']
        st.session_state.df_1["PowerLeft"] = best_solution_data['PowerLeft']["PowerLeft"]
        st.session_state.df_1["PowerReduced"] = best_solution_data['PowerReduced']["PowerReduced"]
        st.session_state.df_1["RUL"] = best_solution_data['RUL']["RUL"]
        
        # Store all results in session state for plotting
        # Clear previous quick results
        keys_to_remove = [k for k in st.session_state.pu_results.keys() if isinstance(k, str) and k.startswith('run_')]
        for k in keys_to_remove:
            del st.session_state.pu_results[k]
        
        # Store all 10 runs
        for run_idx, sol_data in enumerate(all_solutions):
            dts = st.session_state.df_1.copy()
            dts["PU Projection"] = sol_data['solution']
            dts["PowerLeft"] = sol_data['PowerLeft']["PowerLeft"]
            dts["PowerReduced"] = sol_data['PowerReduced']["PowerReduced"]
            dts["RUL"] = sol_data['RUL']["RUL"]
            
            # Store with run identifier
            st.session_state.pu_results[f'run_{run_idx}'] = dts
        
        # Store best result with special key
        dts_best = st.session_state.df_1.copy()
        st.session_state.pu_results['best'] = dts_best
        
        # Latest results
        plot_results()
        
    except Exception as e:
        st.error(f"Error during RL inference: {e}")
        import traceback
        st.error(traceback.format_exc())
        return

def optimisation_sequence(progress_bar=None, force_training=False):
    """Run hierarchical RL optimization - with optional training"""
    
    # Only optimise track with no actual PU results
    df_temp = st.session_state.df_1.copy()
    tracks_to_optimize_idx = df_temp[df_temp["PU Actual"].isna()].index.tolist()
    
    if len(tracks_to_optimize_idx) == 0:
        st.error("No races to optimize")
        return
    
    # Check if pre-trained models exist and user wants quick mode
    models_exist = os.path.exists("models/rl_agents/manager.pth")
    
    if models_exist and not force_training:
        # Use quick inference mode
        quick_optimisation(progress_bar=progress_bar)
        return
    
    # Otherwise, do training
    st.session_state.pu_iter = 0
    st.session_state.pu_fitness_trace = []
    st.session_state.pu_results = {}
    
    # Create RL environment
    env = PUSelectionEnv(
        track_data=df_temp,
        damage_model_func=DamageModel,
        max_pu_usage=3
    )
    
    # Create hierarchical RL coordinator
    coordinator = HierarchicalRLCoordinator(
        env=env,
        damage_model_func=DamageModel,
        num_episodes=st.session_state.num_episodes,
        use_pretrained=models_exist  # Use pre-trained if available
    )
    
    st.session_state.rl_coordinator = coordinator
    
    # Train the agents (or fine-tune)
    try:
        # Create a wrapper callback that includes progress bar
        def wrapped_callback(episode, step, reward):
            progress_callback(episode, step, reward, progress_bar)
        
        coordinator.train(progress_callback=wrapped_callback, fast_mode=True)
    except Exception as e:
        st.error(f"Error during RL training: {e}")
        import traceback
        st.error(traceback.format_exc())
        return
    
    # Get best solution
    best_solution = coordinator.get_best_solution()
    solution = make_full_solution(best_solution)
    
    # Calculate final metrics
    Fitness, PowerLoss, PowerLeft, RUL, PowerReduced = DamageModel(solution)

    st.session_state.df_1["PU Projection"] = solution
    st.session_state.df_1["PowerLeft"] = PowerLeft["PowerLeft"]
    st.session_state.df_1["PowerReduced"] = PowerReduced["PowerReduced"]
    st.session_state.df_1["RUL"] = RUL["RUL"]
    dts = st.session_state.df_1.copy()

    # Store final result
    st.session_state.pu_results[st.session_state.num_episodes] = dts

    # Update plot with final results
    if len(st.session_state.pu_fitness_trace) > 0:
        dta = pd.DataFrame(st.session_state.pu_fitness_trace,columns=["value"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=dta['value'],
            mode='markers+lines',
        ))
        fig.update_yaxes(title_text='Episode Reward')
        fig.update_xaxes(title_text='Episode')
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.session_state.pu_iter_placeholder.plotly_chart(fig,use_container_width=True)

        # Latest results
        plot_results()
    
def parse_fresh_pu_request(prompt):
    """Parse Fresh PU assignment requests from natural language"""
    import re
    prompt_lower = prompt.lower()
    
    # Try to extract PU number and race information
    # Patterns: "set fresh pu 2 for race 5", "assign pu 1 to monaco", "use fresh pu 3 at monza"
    
    # Pattern 1: "fresh pu X for race Y" or "pu X for race Y"
    pattern1 = r"(?:fresh\s+)?pu\s+(\d+)\s+(?:for|to|at)\s+race\s+(\d+)"
    match = re.search(pattern1, prompt_lower)
    if match:
        pu_num = int(match.group(1))
        race_num = int(match.group(2))
        if 1 <= pu_num <= 3 and 1 <= race_num <= 24:
            return {"pu": pu_num, "race_idx": race_num - 1}  # race_idx is 0-based
    
    # Pattern 2: "fresh pu X for [track name]"
    pattern2 = r"(?:fresh\s+)?pu\s+(\d+)\s+(?:for|at|to)\s+(\w+)"
    match = re.search(pattern2, prompt_lower)
    if match:
        pu_num = int(match.group(1))
        track_name = match.group(2)
        if 1 <= pu_num <= 3:
            # Try to find the track in the dataframe
            df = st.session_state.df_1
            for idx, row in df.iterrows():
                if track_name in row['Track'].lower():
                    return {"pu": pu_num, "race_idx": idx}
    
    # Pattern 3: "race X with fresh pu Y" or "race X use pu Y"
    pattern3 = r"race\s+(\d+)\s+(?:with|use|using)?\s*(?:fresh\s+)?pu\s+(\d+)"
    match = re.search(pattern3, prompt_lower)
    if match:
        race_num = int(match.group(1))
        pu_num = int(match.group(2))
        if 1 <= pu_num <= 3 and 1 <= race_num <= 24:
            return {"pu": pu_num, "race_idx": race_num - 1}
    
    return None

def apply_fresh_pu(pu_num, race_idx):
    """Apply Fresh PU assignment to the dataframe and trigger re-optimization"""
    df = st.session_state.df_1.copy()
    df.loc[race_idx, "Fresh PU"] = pu_num
    st.session_state.df_1 = df
    
    # Trigger re-optimization
    progress_bar = st.session_state.get('progress_bar', None)
    quick_optimisation(progress_bar=progress_bar)
    
    # Get track name for confirmation message
    track_name = df.loc[race_idx, 'Track']
    return track_name

def reoptimise():
    
    df = st.session_state.df_1.copy()
    mode_requested = []
    flag = False
    for index, updates in st.session_state["mastertable"].items():
        if "edited_rows" in index:
            for row, value in updates.items():
                for colname, cellvalue in value.items():
                    mode_requested.append(colname)
                    df.loc[row,colname] = cellvalue

                    # Only get update flag for actual manipulation not failure
                    # Flag is for rerunning the decision engine
                    if "PU Actual" in colname:
                        if df.loc[row,colname] == df.loc[row,"PU Projection"]:
                            flag = False
                        else:
                            flag = True


    st.session_state.df_1 = df

    # Get progress bar from session state if available
    progress_bar = st.session_state.get('progress_bar', None)

    if "PU Actual" in mode_requested:
        if flag:
            quick_optimisation(progress_bar=progress_bar)  # Use quick mode for manual changes
            status_placeholder.success('PU allocation is successful', icon="✅")
            time.sleep(0.5)

    if "PU Failures" in mode_requested:
        quick_optimisation(progress_bar=progress_bar)  # Use quick mode for manual changes
        status_placeholder.success('PU allocation is successful', icon="✅")
        time.sleep(0.5)
    
    if "Fresh PU" in mode_requested:
        quick_optimisation(progress_bar=progress_bar)  # Use quick mode for manual changes
        status_placeholder.success('PU allocation is successful', icon="✅")
        time.sleep(0.5)


#---------- Load in track information -------------
   
if not isinstance(st.session_state.df_1,pd.DataFrame):
    
    # Initial condition
    try:
        # Try CSV first (faster and avoids Excel reading issues)
        try:
            df = pd.read_csv('data/Page1_track.csv')
        except FileNotFoundError:
            # Fallback to Excel if CSV doesn't exist
            df = pd.read_excel('data/Page1_track.xlsx')
    except Exception as e:
        st.error(f"Error reading data file: {e}")
        st.stop()
    
    st.session_state.df_1 = df.copy()
    df["Fresh PU"] = np.nan  # User can assign specific PU (1, 2, or 3)
    df["PU Failures"] = np.nan
    df["PU Actual"] = np.nan
    df["PU Projection"] = np.nan
    
    Fitness, PowerLoss, PowerLeft, RUL, PowerReduced = DamageModel(df["PU Projection"].to_numpy())
    df["PowerLeft"] = PowerLeft["PowerLeft"]
    df["PowerReduced"] = PowerReduced["PowerReduced"]
    df["RUL"] = RUL["RUL"]

    df.insert(2, "Fresh PU", df.pop("Fresh PU"))
    df.insert(3, "PU Projection", df.pop("PU Projection"))
    df.insert(3, "PU Actual", df.pop("PU Actual"))
    df.insert(3, "PU Failures", df.pop("PU Failures"))
    df.pop('Date')
    df.pop('No')

    # Going to be the main table
    st.session_state.df_1 = df.copy()

else:

    # Simulate and repopulate the master table
    df = st.session_state.df_1.copy()
    
    # Backward compatibility: rename Assign PU to Fresh PU or add if doesn't exist
    if "Assign PU" in df.columns:
        df.rename(columns={"Assign PU": "Fresh PU"}, inplace=True)
    elif "Fresh PU" not in df.columns:
        df.insert(2, "Fresh PU", np.nan)
    else:
        # If Fresh PU exists but is boolean (old format), convert to NaN
        if df["Fresh PU"].dtype == bool:
            df["Fresh PU"] = np.nan
    
    Fitness, PowerLoss, PowerLeft, RUL, PowerReduced = DamageModel(df["PU Projection"].to_numpy())
    df["PowerLeft"] = PowerLeft["PowerLeft"]
    df["PowerReduced"] = PowerReduced["PowerReduced"]
    df["RUL"] = RUL["RUL"]
    
    # Going to be the main table
    st.session_state.df_1 = df

# ----------- prompt engine ----------------------
def send_message(messages, persona_content):
    persona = [{"role":"system", "content":persona_content}]
    persona.extend(messages)
    openai_response = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[{"role": m["role"], "content": m["content"]} for m in persona],
        stream=True)
    
    return openai_response

def chat_callback():
    prompt = st.session_state['chat_input']
    st.session_state.chat_prompt = prompt

def get_agent_persona(agent_name):
    """Get the persona description for the selected agent"""
    # Add redirect instruction for co-workers
    redirect_instruction = ""
    if agent_name in ["Performance", "Reliability"]:
        redirect_instruction = "\n\nIMPORTANT: You only communicate with the Chief engineer, not directly with users. If a user tries to chat with you directly, politely redirect them to speak with the Chief engineer, as he is your manager and coordinates all communications."
    
    # Get current bias display value
    current_bias = st.session_state.get('pu_bias', 2)
    if current_bias == 1:
        bias_display = "High Performance (1)"
    elif current_bias == 10:
        bias_display = "Longer RUL (10)"
    else:
        bias_display = str(current_bias)
    
    manager_settings_desc = f"""You can modify Manager RL agent settings:
- performance_weight (0.0-1.0): Weight given to Performance agent recommendations (current: {st.session_state.rl_manager_settings['performance_weight']})
- reliability_weight (0.0-1.0): Weight given to Reliability agent recommendations (current: {st.session_state.rl_manager_settings['reliability_weight']})
- max_pu_usage (1-3): Maximum number of PUs that can be used (current: {st.session_state.rl_manager_settings['max_pu_usage']})
- learning_rate (0.0001-0.01): Learning rate for training (current: {st.session_state.rl_manager_settings['learning_rate']})
- gamma (0.9-0.99): Discount factor for future rewards (current: {st.session_state.rl_manager_settings['gamma']})
- num_episodes (5-100): Number of training episodes (current: {st.session_state.get('num_episodes', 20)})
- pu_bias (1-10 or "High Performance"/"Longer RUL"): Optimization bias toward performance (1) or RUL (10) (current: {bias_display})"""
    
    performance_settings_desc = f"""You can modify Performance RL agent settings:
- learning_rate (0.0001-0.01): Learning rate for training (current: {st.session_state.rl_performance_settings['learning_rate']})
- gamma (0.9-0.99): Discount factor for future rewards (current: {st.session_state.rl_performance_settings['gamma']})
- power_reward_scale (1.0-20.0): Reward scaling for selecting high-power PUs (current: {st.session_state.rl_performance_settings['power_reward_scale']})
- power_reduced_penalty_scale (1.0-50.0): Penalty scaling for power degradation (current: {st.session_state.rl_performance_settings['power_reduced_penalty_scale']})"""
    
    reliability_settings_desc = f"""You can modify Reliability RL agent settings:
- learning_rate (0.0001-0.01): Learning rate for training (current: {st.session_state.rl_reliability_settings['learning_rate']})
- gamma (0.9-0.99): Discount factor for future rewards (current: {st.session_state.rl_reliability_settings['gamma']})
- rul_reward_scale (10.0-50.0): Reward scaling for selecting high RUL PUs (current: {st.session_state.rl_reliability_settings['rul_reward_scale']})
- rul_threshold_critical (0.01-0.10): RUL threshold for critical warnings (current: {st.session_state.rl_reliability_settings['rul_threshold_critical']})
- rul_threshold_warning (0.05-0.20): RUL threshold for warnings (current: {st.session_state.rl_reliability_settings['rul_threshold_warning']})
- failure_penalty (-5000 to -500): Penalty for PU failures (current: {st.session_state.rl_reliability_settings['failure_penalty']})"""
    
    personas = {
        "Chief engineer": f"""You are a Chief engineer in a hierarchical RL system for F1 power unit selection. Your role is to oversee rules and PU usage constraints. You coordinate between Performance and Reliability agents to ensure optimal PU allocation throughout the season. You autonomously decide when to consult the Performance agent (for questions about power, degradation, performance optimization) or the Reliability agent (for questions about PU survival, SOH, durability, failure prevention). You can consult both agents if needed, or answer directly if the question doesn't require their expertise.

{manager_settings_desc}

When users request setting changes, parse the request and respond with confirmation. Use phrases like "increase performance weight", "set learning rate to 0.005", "prioritize performance more", etc.""",
        "Performance": f"""You are a Performance agent in a hierarchical RL system for F1 power unit selection. Your primary goal is to minimize degradation and maximize power output. You work alongside the Reliability agent and report to the Chief engineer. You only communicate with the Chief engineer, not directly with users.

{performance_settings_desc}

When the Chief engineer requests setting changes for your RL agent, parse the request and respond with confirmation.

{redirect_instruction}""",
        "Reliability": f"""You are a Reliability agent in a hierarchical RL system for F1 power unit selection. Your primary goal is to ensure SOH (State of Health) > 0 and that all power units survive until the end of the season. You work alongside the Performance agent and report to the Chief engineer. You only communicate with the Chief engineer, not directly with users.

{reliability_settings_desc}

When the Chief engineer requests setting changes for your RL agent, parse the request and respond with confirmation.

{redirect_instruction}"""
    }
    return personas.get(agent_name, personas["Chief engineer"])

def parse_setting_change(prompt, agent_name):
    """Parse setting change requests from natural language"""
    # Get current settings based on agent
    if agent_name == "Chief engineer":
        settings = st.session_state.rl_manager_settings
        setting_names = {
            "performance_weight": ["performance weight", "performance bias", "perf weight", "perf bias"],
            "reliability_weight": ["reliability weight", "reliability bias", "rel weight", "rel bias"],
            "max_pu_usage": ["max pu", "maximum pu", "max power unit", "pu limit"],
            "learning_rate": ["learning rate", "lr", "learning speed"],
            "gamma": ["gamma", "discount", "discount factor"],
            "num_episodes": ["number of episodes", "episodes", "num episodes", "training episodes"],
            "pu_bias": ["bias", "pu bias", "performance bias", "rul bias", "optimization bias"]
        }
    elif agent_name == "Performance":
        settings = st.session_state.rl_performance_settings
        setting_names = {
            "learning_rate": ["learning rate", "lr", "learning speed", "learn rate"],
            "gamma": ["gamma", "discount", "discount factor"],
            "power_reward_scale": ["power reward", "power scale", "power reward scale", "reward scale", "power reward scale"],
            "power_reduced_penalty_scale": ["power penalty", "degradation penalty", "power reduced penalty", "penalty scale", "degradation"]
        }
    elif agent_name == "Reliability":
        settings = st.session_state.rl_reliability_settings
        setting_names = {
            "learning_rate": ["learning rate", "lr", "learning speed"],
            "gamma": ["gamma", "discount", "discount factor"],
            "rul_reward_scale": ["rul reward", "rul scale", "rul reward scale"],
            "rul_threshold_critical": ["critical threshold", "critical rul", "rul critical"],
            "rul_threshold_warning": ["warning threshold", "warning rul", "rul warning"],
            "failure_penalty": ["failure penalty", "failure punishment"]
        }
    else:
        return None
    
    prompt_lower = prompt.lower()
    changes = {}
    
    # Sort aliases by length (longest first) to match more specific phrases first
    sorted_settings = []
    for setting_key, aliases in setting_names.items():
        sorted_aliases = sorted(aliases, key=len, reverse=True)
        sorted_settings.append((setting_key, sorted_aliases))
    
    # Try to extract setting name and value
    import re
    for setting_key, aliases in sorted_settings:
        for alias in aliases:
            if alias in prompt_lower:
                # Try to extract numeric value - look for numbers near the setting name
                # Pattern 1: "setting to X" or "setting = X" or "setting: X" or "setting X"
                patterns = [
                    rf"{re.escape(alias)}\s*(?:to|is|as|=\s*|:\s*|of\s*)\s*(\d+\.?\d*)",
                    rf"{re.escape(alias)}\s+(\d+\.?\d*)",  # "learning rate 0.005"
                    rf"{re.escape(alias)}.*?(\d+\.?\d*)",  # "learning rate something 0.005"
                ]
                value = None
                for pattern in patterns:
                    match = re.search(pattern, prompt_lower)
                    if match:
                        try:
                            value = float(match.group(1))
                            # Handle percentage values
                            if "percent" in prompt_lower or "%" in prompt:
                                value = value / 100.0
                            changes[setting_key] = value
                            break
                        except (ValueError, IndexError):
                            continue
                
                if value is not None:
                    break
                
                # Also try to find any number in the prompt if the setting name is mentioned
                # This handles cases like "set learning rate for performance to 0.005"
                if not value:
                    # Look for any number after the alias
                    number_pattern = r"(\d+\.?\d*)"
                    # Find position of alias
                    alias_pos = prompt_lower.find(alias)
                    if alias_pos >= 0:
                        # Look for numbers after the alias
                        after_alias = prompt_lower[alias_pos + len(alias):]
                        number_match = re.search(number_pattern, after_alias)
                        if number_match:
                            try:
                                value = float(number_match.group(1))
                                if "percent" in prompt_lower or "%" in prompt:
                                    value = value / 100.0
                                changes[setting_key] = value
                                break
                            except (ValueError, IndexError):
                                pass
                
                if value is not None:
                    break
                    
                # Check for relative changes
                if "increase" in prompt_lower or "higher" in prompt_lower or "more" in prompt_lower:
                    current = settings.get(setting_key, 0)
                    if "performance_weight" in setting_key or "reliability_weight" in setting_key:
                        changes[setting_key] = min(1.0, current + 0.1)
                    else:
                        changes[setting_key] = current * 1.2
                    break
                elif "decrease" in prompt_lower or "lower" in prompt_lower or "less" in prompt_lower:
                    current = settings.get(setting_key, 0)
                    if "performance_weight" in setting_key or "reliability_weight" in setting_key:
                        changes[setting_key] = max(0.0, current - 0.1)
                    else:
                        changes[setting_key] = current * 0.8
                    break
                # Special handling for bias requests
                elif "prioritize performance" in prompt_lower or "more performance" in prompt_lower:
                    if "performance_weight" in setting_names:
                        changes["performance_weight"] = min(1.0, settings.get("performance_weight", 0.3) + 0.2)
                        changes["reliability_weight"] = max(0.0, settings.get("reliability_weight", 0.3) - 0.1)
                elif "prioritize reliability" in prompt_lower or "more reliability" in prompt_lower:
                    if "reliability_weight" in setting_names:
                        changes["reliability_weight"] = min(1.0, settings.get("reliability_weight", 0.3) + 0.2)
                        changes["performance_weight"] = max(0.0, settings.get("performance_weight", 0.3) - 0.1)
                
                # Special handling for bias setting (pu_bias)
                if setting_key == "pu_bias":
                    # Check for text values
                    if "high performance" in prompt_lower:
                        changes["pu_bias"] = "High Performance"
                    elif "longer rul" in prompt_lower or "long rul" in prompt_lower:
                        changes["pu_bias"] = "Longer RUL"
                    # If we found a number, it will be handled by the normal parsing above
    
    # Special handling for bias text values (can be mentioned without "bias" keyword)
    if "high performance" in prompt_lower and ("bias" in prompt_lower or "set" in prompt_lower or "change" in prompt_lower):
        changes["pu_bias"] = "High Performance"
    elif ("longer rul" in prompt_lower or "long rul" in prompt_lower) and ("bias" in prompt_lower or "set" in prompt_lower or "change" in prompt_lower):
        changes["pu_bias"] = "Longer RUL"
    
    # If we found a setting name but no value, try to find any number in the prompt
    # This is a last resort fallback
    if not changes:
        # Look for any setting name mentioned
        for setting_key, aliases in sorted_settings:
            for alias in aliases:
                if alias in prompt_lower:
                    # Find any number in the entire prompt
                    numbers = re.findall(r'\d+\.?\d*', prompt_lower)
                    if numbers:
                        try:
                            value = float(numbers[0])  # Use first number found
                            if "percent" in prompt_lower or "%" in prompt:
                                value = value / 100.0
                            changes[setting_key] = value
                            break
                        except (ValueError, IndexError):
                            pass
            if changes:
                break
    
    return changes if changes else None

def apply_rl_settings(agent_name, settings_dict):
    """Apply settings to the appropriate RL agent settings"""
    if agent_name == "Chief engineer":
        for key, value in settings_dict.items():
            if key in st.session_state.rl_manager_settings:
                # Validate ranges
                if key in ["performance_weight", "reliability_weight"]:
                    value = max(0.0, min(1.0, value))
                elif key == "max_pu_usage":
                    value = max(1, min(3, int(value)))
                elif key == "learning_rate":
                    value = max(0.0001, min(0.01, value))
                elif key == "gamma":
                    value = max(0.9, min(0.99, value))
                st.session_state.rl_manager_settings[key] = value
            elif key == "num_episodes":
                # Handle number of episodes
                value = max(5, min(100, int(value)))
                st.session_state.num_episodes = value
                # The number_input will pick up the new value on next rerun since it reads from session_state
            elif key == "pu_bias":
                # Handle bias - can be numeric (1-10) or text (High Performance, Longer RUL)
                if isinstance(value, str):
                    value_lower = value.lower()
                    if "high performance" in value_lower or value_lower == "1":
                        st.session_state.pu_bias = 1
                        # Update slider value
                        st.session_state.slider = 'High Performance'
                    elif "longer rul" in value_lower or "long rul" in value_lower or value_lower == "10":
                        st.session_state.pu_bias = 10
                        st.session_state.slider = 'Longer RUL'
                    else:
                        # Try to parse as number
                        try:
                            bias_num = int(float(value))
                            bias_num = max(1, min(10, bias_num))
                            st.session_state.pu_bias = bias_num
                            st.session_state.slider = str(bias_num)
                        except (ValueError, TypeError):
                            pass
                else:
                    # Numeric value
                    bias_num = max(1, min(10, int(value)))
                    st.session_state.pu_bias = bias_num
                    if bias_num == 1:
                        st.session_state.slider = 'High Performance'
                    elif bias_num == 10:
                        st.session_state.slider = 'Longer RUL'
                    else:
                        st.session_state.slider = str(bias_num)
                # Set flag to trigger re-optimization
                st.session_state.bias_changed = True
                # Clear the slider widget's state so it picks up the new value
                if 'slider' in st.session_state:
                    # The slider key is used by the widget, we need to update it
                    # The value is already set above, so we just need to ensure rerun happens
                    pass
    elif agent_name == "Performance":
        for key, value in settings_dict.items():
            if key in st.session_state.rl_performance_settings:
                if key == "learning_rate":
                    value = max(0.0001, min(0.01, value))
                elif key == "gamma":
                    value = max(0.9, min(0.99, value))
                elif key == "power_reward_scale":
                    value = max(1.0, min(20.0, value))
                elif key == "power_reduced_penalty_scale":
                    value = max(1.0, min(50.0, value))
                st.session_state.rl_performance_settings[key] = value
    elif agent_name == "Reliability":
        for key, value in settings_dict.items():
            if key in st.session_state.rl_reliability_settings:
                if key == "learning_rate":
                    value = max(0.0001, min(0.01, value))
                elif key == "gamma":
                    value = max(0.9, min(0.99, value))
                elif "threshold" in key:
                    value = max(0.01, min(0.5, value))
                st.session_state.rl_reliability_settings[key] = value

def determine_coworker_to_consult(prompt, chief_messages):
    """Use Chief engineer to determine which co-worker(s) to consult"""
    
    # Rule-based pre-check: Don't consult agents for setting/configuration questions
    prompt_lower = prompt.lower()
    setting_keywords = [
        "change", "set", "update", "modify", "configure",
        "episodes", "bias", "learning rate", "gamma", "weight",
        "settings", "parameter", "value", "number of",
        "how does", "what is", "explain", "tell me about",
        "current", "status", "show me",
        "fresh pu", "fresh power unit", "assign pu", "set pu for race"
    ]
    
    # If it's clearly a setting/configuration question, don't consult agents
    if any(keyword in prompt_lower for keyword in setting_keywords):
        # Exception: if asking about performance/reliability strategies, still consult
        strategy_keywords = ["strategy", "optimize", "best way to", "how to improve", "technique", "should i"]
        if not any(sk in prompt_lower for sk in strategy_keywords):
            return []  # Chief Engineer handles directly
    
    decision_prompt = f"""Based on this user question: "{prompt}", determine which co-worker agent(s) should be consulted:

CONSULT "Performance" ONLY if the question is specifically about:
- Performance optimization strategies
- Power degradation analysis
- How to maximize power output
- Performance-related technical decisions

CONSULT "Reliability" ONLY if the question is specifically about:
- Reliability strategies
- SOH (State of Health) management
- PU failure prevention techniques
- Durability analysis

CONSULT "Both" ONLY if the question requires technical input from both domains.

ANSWER "None" for questions about:
- Changing settings (episodes, bias, weights, learning rate, etc.)
- Setting Fresh PU for races
- Assigning PUs to races
- General system operation
- How the RL system works
- Status inquiries
- Configuration changes
- Operational tasks
- Anything you can answer directly as Chief Engineer

Respond with ONLY one word: "Performance", "Reliability", "Both", or "None". Do not include any explanation."""
    
    decision_messages = chief_messages.copy()
    decision_messages.append({"role": "user", "content": decision_prompt})
    
    persona = "You are a Chief engineer making a decision about which co-worker to consult."
    openai_response = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[{"role": "system", "content": persona}] + [{"role": m["role"], "content": m["content"]} for m in decision_messages],
        temperature=0.3  # Lower temperature for more deterministic decisions
    )
    
    decision = openai_response.choices[0].message.content.strip().lower()
    
    if "both" in decision:
        return ["Performance", "Reliability"]
    elif "performance" in decision:
        return ["Performance"]
    elif "reliability" in decision:
        return ["Reliability"]
    else:
        return []

def get_agent_icon(agent_name):
    """Get the icon for the agent"""
    icons = {
        "Chief engineer": "👔",
        "Performance": "⚡",
        "Reliability": "🛡️"
    }
    return icons.get(agent_name, "🤖")



# ----------- UI Section -------------------------
    
st_title('PU Decision Engine Playground')

st.caption("Optimized for dark mode. To change the theme, access the settings panel by clicking the three dots in the top-right corner of the app.")

with st.expander('Introduction',expanded=True):

    # Display both images side by side
    col1, col2 = st.columns([1,0.7])
    
    with col1:

        st_text('A virtual environment to demonstrate the ability of Hierarchical Multi-Agent Reinforcement Learning to solve PU selection problem. The system consists of a Manager agent that oversees rules and PU usage constraints, and two co-worker agents: Performance (minimizes degradation) and Reliability (ensures PU alive until end of the season). Allows race engineer to quickly restrategise live with new race data and historical decisions.')
    
    with col2:
        st_text('The system consists of a Manager agent that oversees rules and PU usage constraints, and two co-worker agents: Performance (minimizes degradation) and Reliability (ensures SOH > 0). Allows race engineer to quickly restrategise live with new race data and historical decisions. Adapted from Farraen\'s 2018 Matlab GA PU script and converted into Python environment with RL optimization. The UI was developed using 2018 season track data.')

    # Display both images side by side
    col1, col2 = st.columns([1,0.7])
    
    with col1:

        image_intro = read_image("images/Page1_intro.png")
        st.image(image_intro, use_column_width=True, caption="PU Selection Overview")
    
    with col2:
        image_rl = read_image("images/rl.png")
        st.image(image_rl, use_column_width=True, caption="Reinforcement Learning System")


    st.write("Comparison between optimisers")
    image_comparison = read_image("images/rl_2.png")
    st.image(image_comparison, use_column_width=True)



with st.expander('Damage model',expanded=False):
    st_text('The optimiser uses an artificial damage model made solely for demonstration purposes. The data does not represent true PU values.')
    image = read_image("images/Page1_damage.png")
    st.image(image,width=700,use_column_width=True)

with st.expander('AI race engineer - Chat with RL agents',expanded=True):

    st_text("Still work in progress to allow discussion with the RL agents. For now, you can only chat with the Chief engineer.")
    # User only chats with Chief engineer
    st.markdown("### 👔 Chief Engineer")
    st.caption("Oversees rules and PU usage constraints. Automatically consults with Performance and Reliability agents when needed.")
    
    # Chat input - user only talks to Chief engineer
    chat_input = st.chat_input("Enter your message to Chief Engineer:")
    if chat_input:
        st.session_state.chat_prompt = chat_input
        st.rerun()
    
    # Display chat history with full conversation flow
    for message in st.session_state.chat_messages:
        if "user" in message["role"]:
            st.write(f"🧑    {message['content']}")
        else:
            # Display agent icon and name for assistant messages
            agent_name = message.get("agent", "Assistant")
            agent_icon = get_agent_icon(agent_name)
            
            # Show if message was forwarded from/to another agent
            forwarded_from = message.get("forwarded_from")
            forwarded_to = message.get("forwarded_to", [])
            internal_note = message.get("internal_note", "")
            
            if isinstance(forwarded_to, str):
                forwarded_to = [forwarded_to]
            
            # Format message display based on context
            if internal_note and "consulting" in internal_note.lower():
                # Chief engineer consulting co-worker - show the forward message
                coworker = forwarded_to[0] if forwarded_to else "co-worker"
                st.write(f"{agent_icon} **{agent_name}** → {coworker}: :blue[{message['content']}]")
            elif forwarded_from:
                # Co-worker responding to Chief engineer
                st.write(f"{agent_icon} **{agent_name}** → {forwarded_from}: :blue[{message['content']}]")
            elif forwarded_to:
                # Message being forwarded to co-workers
                st.write(f"{agent_icon} **{agent_name}** (consulting {', '.join(forwarded_to)}): :blue[{message['content']}]")
            else:
                # Direct message from agent
                st.write(f"{agent_icon} **{agent_name}**: :blue[{message['content']}]")
    
    # Process new prompt
    if st.session_state.chat_prompt:
        prompt = st.session_state.chat_prompt
        current_agent = "Chief engineer"  # User always talks to Chief engineer
        
        # Debug: Show that we're processing the prompt
        # st.write(f"🔧 DEBUG: Processing prompt: '{prompt}'")
        
        # Check if user is trying to chat directly with co-workers
        prompt_lower = prompt.lower()
        direct_co_worker_mention = False
        mentioned_co_worker = None
        
        if ("performance agent" in prompt_lower or "performance co-worker" in prompt_lower or 
            "hey performance" in prompt_lower or "performance," in prompt_lower):
            direct_co_worker_mention = True
            mentioned_co_worker = "Performance"
        elif ("reliability agent" in prompt_lower or "reliability co-worker" in prompt_lower or 
              "hey reliability" in prompt_lower or "reliability," in prompt_lower):
            direct_co_worker_mention = True
            mentioned_co_worker = "Reliability"
        
        # Display user message
        st.write(f"🧑   {prompt}")
        
        # Add user message to chat history
        st.session_state.chat_messages.append({
            "role": "user", 
            "content": prompt,
            "agent": current_agent
        })
        
        # If user tried to chat directly with co-worker, have Chief engineer redirect them
        if direct_co_worker_mention and mentioned_co_worker:
            redirect_message = f"I notice you're trying to communicate with the {mentioned_co_worker} agent. As the Chief engineer, I coordinate all communications with my team. Please direct your questions to me, and I will consult with the {mentioned_co_worker} agent if needed. How can I help you?"
            
            st.write(f"👔 **Chief engineer**: :blue[{redirect_message}]")
            
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": redirect_message,
                "agent": current_agent
            })
            
            st.session_state.chat_messages = st.session_state.chat_messages[-20:]
            st.session_state.chat_prompt = []
            st.rerun()
        else:
            # Check if this is a Fresh PU request
            fresh_pu_request = parse_fresh_pu_request(prompt)
            if fresh_pu_request:
                # Apply the Fresh PU assignment
                track_name = apply_fresh_pu(fresh_pu_request['pu'], fresh_pu_request['race_idx'])
                success_message = f"✅ Fresh PU {fresh_pu_request['pu']} has been assigned to {track_name} (Race {fresh_pu_request['race_idx'] + 1}). The table and optimization have been updated."
                st.success(success_message)
                
                # Add confirmation to chat
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": success_message,
                    "agent": current_agent
                })
                
                st.session_state.chat_messages = st.session_state.chat_messages[-20:]
                st.session_state.chat_prompt = []
                st.rerun()
            
            # Check if this is a setting change request
            setting_changes = parse_setting_change(prompt, current_agent)
            setting_change_applied = False
            
            # Check if user wants to change co-worker settings (regardless of Chief Engineer settings)
            prompt_lower = prompt.lower()
            is_coworker_setting_request = False
            target_coworker = None
            
            # Detect if this is a request to change Performance or Reliability agent settings
            # Look for keywords that indicate setting changes
            setting_keywords = ["change", "update", "modify", "set", "increase", "decrease", "adjust", "setting", "configure"]
            
            # More flexible detection for Performance agent settings
            # Check for Performance agent/engineer mentions
            if "performance" in prompt_lower:
                # Check if it's about Performance agent settings
                performance_indicators = [
                    "performance agent", "performance engineer", 
                    "tell performance", "for performance",
                    "performance's", "performance agent's", "performance engineer's"
                ]
                # Also check if any Performance-specific settings are mentioned
                perf_setting_names = ["power reward", "power scale", "power penalty", "degradation penalty"]
                has_perf_setting = any(ps in prompt_lower for ps in perf_setting_names)
                
                if (any(keyword in prompt_lower for keyword in setting_keywords) or 
                    any(indicator in prompt_lower for indicator in performance_indicators) or
                    has_perf_setting):
                    is_coworker_setting_request = True
                    target_coworker = "Performance"
            elif "reliability" in prompt_lower:
                # Check if it's about Reliability agent settings
                reliability_indicators = [
                    "reliability agent", "reliability engineer", 
                    "tell reliability", "for reliability",
                    "reliability's", "reliability agent's", "reliability engineer's"
                ]
                # Also check if any Reliability-specific settings are mentioned
                rel_setting_names = ["rul reward", "rul scale", "failure penalty", "rul threshold"]
                has_rel_setting = any(rs in prompt_lower for rs in rel_setting_names)
                
                if (any(keyword in prompt_lower for keyword in setting_keywords) or 
                    any(indicator in prompt_lower for indicator in reliability_indicators) or
                    has_rel_setting):
                    is_coworker_setting_request = True
                    target_coworker = "Reliability"
            
            if setting_changes:
                # Apply setting changes for Chief engineer
                apply_rl_settings(current_agent, setting_changes)
                setting_change_applied = True
            
            # Determine which co-workers to consult
            if is_coworker_setting_request and target_coworker:
                # Forward to the specific co-worker for setting changes
                coworkers_to_consult = [target_coworker]
                # Debug: show that we detected a coworker setting request
                st.session_state[f'_debug_coworker_request_{target_coworker}'] = prompt
                # Show debug info to user - use st.write to ensure it's visible
                st.write(f"🔍 **Debug**: Detected {target_coworker} agent setting change request")
                st.info(f"🔍 Detected {target_coworker} agent setting change request")
            else:
                # Get messages for Chief engineer
                chief_messages = []
                for msg in st.session_state.chat_messages:
                    if msg.get("role") == "user" and msg.get("agent") == current_agent:
                        chief_messages.append({"role": "user", "content": msg["content"]})
                    elif msg.get("role") == "assistant" and msg.get("agent") == current_agent:
                        chief_messages.append({"role": "assistant", "content": msg["content"]})
                
                # Chief engineer autonomously decides which co-worker(s) to consult
                coworkers_to_consult = determine_coworker_to_consult(prompt, chief_messages)
            
            coworker_responses = {}
            
            # Consult with co-workers if needed
            if coworkers_to_consult:
                for coworker in coworkers_to_consult:
                    # Get co-worker's conversation history with Chief engineer
                    coworker_messages = []
                    for msg in st.session_state.chat_messages:
                        # Include messages where Chief engineer communicated with this co-worker
                        forwarded_to = msg.get("forwarded_to", [])
                        if isinstance(forwarded_to, str):
                            forwarded_to = [forwarded_to]
                        if msg.get("role") == "user" and msg.get("agent") == current_agent and coworker in forwarded_to:
                            coworker_messages.append({"role": "user", "content": msg["content"]})
                        elif msg.get("role") == "assistant" and msg.get("agent") == coworker:
                            coworker_messages.append({"role": "assistant", "content": msg["content"]})
                    
                    # Check if this is a setting change request for the co-worker
                    is_coworker_setting_change = (is_coworker_setting_request and coworker == target_coworker)
                    
                    # Chief engineer forwards the question
                    if is_coworker_setting_change:
                        # This is a setting change request for the co-worker
                        forwarded_prompt = f"The Chief engineer is requesting to change your RL agent settings: '{prompt}'. Please parse the request, update your settings accordingly, and confirm the changes."
                    else:
                        forwarded_prompt = f"The Chief engineer is asking: '{prompt}'. Please provide your expert opinion on this matter."
                    coworker_messages.append({"role": "user", "content": forwarded_prompt})
                    
                    # Add Chief Engineer's message to co-worker as a visible message in chat
                    chief_to_coworker_message = f"→ {coworker}: {forwarded_prompt}"
                    st.write(f"👔 **Chief engineer** {chief_to_coworker_message}")
                    
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": forwarded_prompt,
                        "agent": current_agent,
                        "forwarded_to": [coworker],
                        "internal_note": f"Chief engineer consulting {coworker}"
                    })
                    
                    # Check if co-worker needs to apply setting changes
                    # Always check for coworker setting changes if it's a setting request
                    coworker_setting_changes = None
                    if is_coworker_setting_change:
                        # Parse the original user prompt for setting changes
                        # Clean the prompt to remove "tell performance agent" etc. to help parsing
                        # Only remove prefixes at the beginning of the sentence
                        cleaned_prompt = prompt_lower.strip()
                        # Remove common prefixes that might interfere with parsing (only at start)
                        prefixes_to_remove = [
                            "tell performance agent to", "tell performance to", 
                            "tell reliability agent to", "tell reliability to",
                            "ask performance agent to", "ask performance to",
                            "ask reliability agent to", "ask reliability to",
                            "for performance agent", "for performance",
                            "for reliability agent", "for reliability",
                            "change performance engineer", "change performance agent",
                            "update performance engineer", "update performance agent",
                            "set performance engineer", "set performance agent"
                        ]
                        for prefix in prefixes_to_remove:
                            if cleaned_prompt.startswith(prefix):
                                cleaned_prompt = cleaned_prompt[len(prefix):].strip()
                        
                        # Remove standalone mentions but be careful not to remove too much
                        # Only remove if they're not part of a setting name
                        cleaned_prompt = cleaned_prompt.replace("performance agent's", "").replace("performance engineer's", "")
                        cleaned_prompt = cleaned_prompt.replace("reliability agent's", "").replace("reliability engineer's", "")
                        # Don't remove "performance" or "reliability" as they might be part of setting names
                        # Clean up extra spaces
                        cleaned_prompt = " ".join(cleaned_prompt.split())
                        
                        # Debug: show what we're trying to parse
                        st.write(f"🔧 **Debug**: Trying to parse cleaned prompt: '{cleaned_prompt}'")
                        
                        # Try parsing with cleaned prompt first, then original if that fails
                        coworker_setting_changes = parse_setting_change(cleaned_prompt, coworker)
                        if not coworker_setting_changes:
                            # Fallback to original prompt (case-insensitive)
                            coworker_setting_changes = parse_setting_change(prompt_lower, coworker)
                        if not coworker_setting_changes:
                            # Last fallback: try with original prompt as-is
                            coworker_setting_changes = parse_setting_change(prompt, coworker)
                        
                        if coworker_setting_changes:
                            # Store old values for comparison
                            if coworker == "Performance":
                                old_settings = st.session_state.rl_performance_settings.copy()
                            elif coworker == "Reliability":
                                old_settings = st.session_state.rl_reliability_settings.copy()
                            else:
                                old_settings = {}
                            
                            apply_rl_settings(coworker, coworker_setting_changes)
                            
                            # Verify settings were actually updated
                            if coworker == "Performance":
                                new_settings = st.session_state.rl_performance_settings
                            elif coworker == "Reliability":
                                new_settings = st.session_state.rl_reliability_settings
                            else:
                                new_settings = {}
                            
                            # Settings are now updated in session state - panel will update on next rerun
                            # Add a note to the chat that settings were updated
                            changes_list = [f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in coworker_setting_changes.items()]
                            st.info(f"✅ {coworker} agent settings updated: {', '.join(changes_list)}")
                            
                            # Mark that settings were updated so we can verify in the panel
                            st.session_state[f'{coworker.lower()}_settings_updated'] = True
                            # Also set a flag for the settings panel to show update
                            if coworker == "Performance":
                                st.session_state['performance_settings_updated'] = True
                            elif coworker == "Reliability":
                                st.session_state['reliability_settings_updated'] = True
                        else:
                            # If parsing failed, show a warning with debugging info
                            st.warning(f"⚠️ Could not parse setting changes from your request. Cleaned prompt: '{cleaned_prompt}'. Please be more specific, e.g., 'set learning rate to 0.005' or 'increase power reward scale to 10'")
                    
                    # Get co-worker's response
                    coworker_persona = get_agent_persona(coworker)
                    openai_response = send_message(coworker_messages, coworker_persona)
                    
                    message_placeholder = st.empty()
                    coworker_response = ""
                    for response in openai_response:
                        coworker_response += (response.choices[0].delta.content or "")
                        coworker_icon = get_agent_icon(coworker)
                        message_placeholder.write(f"{coworker_icon} **{coworker}** (via Chief Engineer): :blue[{coworker_response}]▌")
                    
                    coworker_icon = get_agent_icon(coworker)
                    message_placeholder.write(f"{coworker_icon} **{coworker}** → Chief engineer: :blue[{coworker_response}]")
                    
                    # Store co-worker response
                    coworker_responses[coworker] = coworker_response
                    
                    # Add co-worker's response to chat history
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": coworker_response,
                        "agent": coworker,
                        "forwarded_from": current_agent,
                        "internal_note": f"{coworker} responding to Chief engineer"
                    })
                    
                    # Mark the user message as forwarded
                    for i in range(len(st.session_state.chat_messages) - 1, -1, -1):
                        if (st.session_state.chat_messages[i].get("role") == "user" and 
                            st.session_state.chat_messages[i].get("agent") == current_agent and
                            st.session_state.chat_messages[i].get("content") == prompt):
                            if "forwarded_to" not in st.session_state.chat_messages[i]:
                                st.session_state.chat_messages[i]["forwarded_to"] = []
                            if isinstance(st.session_state.chat_messages[i]["forwarded_to"], str):
                                st.session_state.chat_messages[i]["forwarded_to"] = [st.session_state.chat_messages[i]["forwarded_to"]]
                            if coworker not in st.session_state.chat_messages[i]["forwarded_to"]:
                                st.session_state.chat_messages[i]["forwarded_to"].append(coworker)
                            break
            
            # Chief engineer provides final response (with or without co-worker consultation)
            chief_messages = []
            for msg in st.session_state.chat_messages:
                if msg.get("role") == "user" and msg.get("agent") == current_agent:
                    chief_messages.append({"role": "user", "content": msg["content"]})
                elif msg.get("role") == "assistant" and msg.get("agent") == current_agent:
                    chief_messages.append({"role": "assistant", "content": msg["content"]})
            
            # If setting changes were applied, add confirmation to Chief engineer's context
            if setting_changes:
                changes_summary = "Settings updated: " + ", ".join([f"{k}={v}" for k, v in setting_changes.items()])
                chief_messages.append({"role": "system", "content": changes_summary})
            
            if coworker_responses:
                # Build consultation summary
                consultation_summary = "User asked: '" + prompt + "'. "
                for coworker, response in coworker_responses.items():
                    consultation_summary += f"The {coworker} agent responded: '{response}'. "
                consultation_summary += "Please provide a comprehensive response incorporating the co-worker consultations, or answer directly if you have additional insights."
                chief_messages.append({"role": "user", "content": consultation_summary})
            else:
                # Direct response without consultation
                chief_messages.append({"role": "user", "content": prompt})
            
            chief_persona = get_agent_persona("Chief engineer")
            openai_response = send_message(chief_messages, chief_persona)
            
            message_placeholder = st.empty()
            chief_response = ""
            for response in openai_response:
                chief_response += (response.choices[0].delta.content or "")
                agent_icon = get_agent_icon(current_agent)
                message_placeholder.write(f"{agent_icon} **{current_agent}**: :blue[{chief_response}]▌")
            
            agent_icon = get_agent_icon(current_agent)
            message_placeholder.write(f"{agent_icon} **{current_agent}**: :blue[{chief_response}]")
            
            # Add Chief engineer's response to chat history
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": chief_response,
                "agent": current_agent
            })
            
            # Keep last 20 messages total
            st.session_state.chat_messages = st.session_state.chat_messages[-20:]
            st.session_state.chat_prompt = []
            st.rerun()

with st.expander('PU selection optimisation',expanded=True):

    st.write('PU allocation strategy table (Auto-update)')
    st.markdown(
    """
    - :green[Initialise button]: Press to initialise the PU selection using pre-trained RL models. Fills in PU Projection column instantly. For best results, pre-train models first using the button in RL settings.
    - :orange[Fresh PU column]: Assign a specific PU (1, 2, or 3) to a race. The optimizer will ensure that PU is NOT used in any previous races. Use this to force fresh PUs at power tracks.
    - :blue[Actual column]: Use the column with actual selection for completed races and RL will decide the best allocation for the next races. If there is no change, then RL will not trigger.
    - :red[Failure column]: Use the Failure column to exclude failed PU (fill in using PU index 1,2,3,..)
    """
    )

    # Layout: Initialise button on left, bias slider on right
    col_init, col_bias = st.columns([1, 2], gap="medium")
    
    with col_init:
        start_button = st.button('Initialise', use_container_width=True)
        my_bar = st.progress(0)
        st.session_state.progress_bar = my_bar
    
    with col_bias:
        # Get current bias value for slider
        current_slider_value = st.session_state.get('slider', '2')
        # Ensure it's a valid option
        slider_options = ['High Performance','2','3','4','5','6','7','8','9','Longer RUL']
        if current_slider_value not in slider_options:
            # Convert pu_bias to slider value if needed
            pu_bias = st.session_state.get('pu_bias', 2)
            if pu_bias == 1:
                current_slider_value = 'High Performance'
            elif pu_bias == 10:
                current_slider_value = 'Longer RUL'
            else:
                current_slider_value = str(pu_bias)
        
        bias_value = st.select_slider(
            'Select bias:',
            options=slider_options,
            value=current_slider_value,
            on_change=change_bias,
            key='slider')
        
        # Show current bias effect
        if bias_value == 'High Performance':
            st.caption("⚡ Prioritizing Performance (minimize degradation)")
        elif bias_value == 'Longer RUL':
            st.caption("🛡️ Prioritizing Reliability (ensure SOH > 0)")
        else:
            bias_num = int(bias_value)
            perf_pct = int((10 - bias_num) * 10)
            rel_pct = int((bias_num - 1) * 10)
            st.caption(f"⚖️ Balanced: {perf_pct}% Performance, {rel_pct}% Reliability")

    # Highligh rows depending on type (actual or projection)
    df_track_styled = st.session_state.df_1.copy()
    
    # Convert PU columns to integers for display (handle NaN values with nullable integer type)
    pu_columns = ["PU Projection", "PU Actual", "Fresh PU", "PU Failures"]
    for col in pu_columns:
        if col in df_track_styled.columns:
            # Convert to nullable integer type (Int64) to preserve NaN values
            df_track_styled[col] = df_track_styled[col].astype('Int64')

    actual_row = np.where(~df_track_styled["PU Actual"].isna())[0]
    projection_row = np.where(df_track_styled["PU Actual"].isna())[0]

    df_track_styled = df_track_styled.style.set_properties(subset = pd.IndexSlice[actual_row, :], **{'background-color' : 'darkgreen'})\
    .set_properties(subset = pd.IndexSlice[projection_row, :], **{'background-color' : 'midnightblue'})

    # create data editor for the master table
    st.data_editor(df_track_styled,use_container_width=True,key='mastertable',disabled=["PU Projection"],on_change=reoptimise)
    status_placeholder = st.empty()

with st.expander('Optimisation results',expanded=True):
    col1,col2 = st.columns([1,1])
    with col1:
        st.session_state.iterCompare_placeholder = st.empty()
    
    with col2:
        st.session_state.PUCompare_placeholder = st.empty()


with st.expander('Reinforcement Learning settings',expanded=True):

    col1, col2 = st.columns([0.8,1],gap='Small')

    with col1:
        # Use current session state value, default to 20 if not set
        # Read directly from session state so chat updates are reflected
        current_episodes = st.session_state.get('num_episodes', 20)
        # Update session state with the widget value (user can still change it manually)
        new_episodes = st.number_input("Number of episodes (for training)", value=current_episodes, min_value=5, max_value=100, placeholder="Type a number...")
        st.session_state.num_episodes = new_episodes
        
        # Check if pre-trained models exist
        models_exist = os.path.exists("models/rl_agents/manager.pth")
        if models_exist:
            st.success("✓ Pre-trained models available - using quick inference mode")
            
            # Retrain button
            if st.button("🔄 Retrain Models", help="Retrain models on diverse scenarios with current settings", type="secondary"):
                with st.spinner("Retraining RL agents on diverse scenarios... This may take a few minutes."):
                    from pre_train_rl import pre_train_agents
                    try:
                        pre_train_agents(
                            track_data=st.session_state.df_1.copy(),
                            damage_model_func=DamageModel,
                            num_scenarios=10,  # Reduced for speed
                            episodes_per_scenario=8,  # Reduced for speed
                            retrain=True  # Load existing models and continue training
                        )
                        st.success("Retraining complete! Models updated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Retraining error: {e}")
                        import traceback
                        st.error(traceback.format_exc())
        else:
            st.warning("⚠ No pre-trained models - will train on first run")
            if st.button("Pre-train Models (Recommended)", help="Train models on diverse scenarios for better generalization"):
                with st.spinner("Pre-training RL agents on diverse scenarios... This may take a few minutes."):
                    from pre_train_rl import pre_train_agents
                    try:
                        pre_train_agents(
                            track_data=st.session_state.df_1.copy(),
                            damage_model_func=DamageModel,
                            num_scenarios=10,  # Reduced for speed
                            episodes_per_scenario=8  # Reduced for speed
                        )
                        st.success("Pre-training complete! Models saved.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Pre-training error: {e}")
                        import traceback
                        st.error(traceback.format_exc())
        
        st.caption("Hierarchical RL System:")
        st.caption("• Manager: Rules & PU usage constraints")
        st.caption("• Performance Agent: Minimizes degradation")
        st.caption("• Reliability Agent: Ensures SOH > 0")

    with col2:
        st.session_state.pu_iter_placeholder = st.empty()

with st.expander('RL Agent Settings Status (Read-Only)',expanded=False):
    st.caption("Current RL agent settings. These can only be modified through chat with the Chief Engineer. Settings update automatically when changed via chat.")
    
    # Manager (Chief Engineer) Settings
    st.markdown("### 👔 Manager Agent Settings (Chief Engineer)")
    manager_settings = st.session_state.get('rl_manager_settings', {})
    
    # Get current bias and episodes for display
    current_bias = st.session_state.get('pu_bias', 2)
    if current_bias == 1:
        bias_display = "High Performance"
    elif current_bias == 10:
        bias_display = "Longer RUL"
    else:
        bias_display = str(current_bias)
    
    col_m1, col_m2, col_m3 = st.columns(3, gap="small")
    with col_m1:
        st.metric("Performance Weight", f"{manager_settings.get('performance_weight', 0.3):.3f}", 
                 help="Weight given to Performance agent recommendations (0.0-1.0)")
        st.metric("Max PU Usage", manager_settings.get('max_pu_usage', 3),
                 help="Maximum number of PUs that can be used (1-3)")
        st.metric("Number of Episodes", st.session_state.get('num_episodes', 20),
                 help="Number of training episodes (5-100)")
    with col_m2:
        st.metric("Reliability Weight", f"{manager_settings.get('reliability_weight', 0.3):.3f}",
                 help="Weight given to Reliability agent recommendations (0.0-1.0)")
        st.metric("Learning Rate", f"{manager_settings.get('learning_rate', 0.003):.4f}",
                 help="Learning rate for training (0.0001-0.01)")
        st.metric("Optimization Bias", bias_display,
                 help="Bias toward performance (1) or RUL (10)")
    with col_m3:
        st.metric("Gamma", f"{manager_settings.get('gamma', 0.99):.3f}",
                 help="Discount factor for future rewards (0.9-0.99)")
        st.metric("Fresh PU Reward", f"{manager_settings.get('fresh_pu_reward', 10.0):.1f}",
                 help="Reward for following Fresh PU constraints")
    
    st.divider()
    
    # Performance Agent Settings
    st.markdown("### ⚡ Performance Agent Settings")
    # Check if settings were recently updated
    perf_settings_updated = st.session_state.get('performance_settings_updated', False)
    if perf_settings_updated:
        st.success("🔄 Settings were recently updated via chat")
        # Clear the flag after showing
        st.session_state['performance_settings_updated'] = False
    perf_settings = st.session_state.get('rl_performance_settings', {})
    
    # Debug: Show raw settings values (can be removed later)
    # st.caption(f"Debug - Settings dict: {perf_settings}")
    
    col_p1, col_p2, col_p3 = st.columns(3, gap="small")
    with col_p1:
        st.metric("Learning Rate", f"{perf_settings.get('learning_rate', 0.003):.4f}",
                 help="Learning rate for training (0.0001-0.01)")
        st.metric("Power Reward Scale", f"{perf_settings.get('power_reward_scale', 5.0):.1f}",
                 help="Reward scaling for selecting high-power PUs (1.0-20.0)")
    with col_p2:
        st.metric("Gamma", f"{perf_settings.get('gamma', 0.99):.3f}",
                 help="Discount factor for future rewards (0.9-0.99)")
        st.metric("Power Reduced Penalty", f"{perf_settings.get('power_reduced_penalty_scale', 10.0):.1f}",
                 help="Penalty scaling for power degradation (1.0-50.0)")
    with col_p3:
        st.metric("Epsilon", f"{perf_settings.get('epsilon', 1.0):.3f}",
                 help="Exploration rate (1.0-0.01)")
        st.metric("Epsilon Decay", f"{perf_settings.get('epsilon_decay', 0.995):.4f}",
                 help="Epsilon decay rate per episode")
    
    st.divider()
    
    # Reliability Agent Settings
    st.markdown("### 🛡️ Reliability Agent Settings")
    rel_settings = st.session_state.get('rl_reliability_settings', {})
    
    col_r1, col_r2, col_r3 = st.columns(3, gap="small")
    with col_r1:
        st.metric("Learning Rate", f"{rel_settings.get('learning_rate', 0.003):.4f}",
                 help="Learning rate for training (0.0001-0.01)")
        st.metric("RUL Reward Scale", f"{rel_settings.get('rul_reward_scale', 30.0):.1f}",
                 help="Reward scaling for selecting high RUL PUs (10.0-50.0)")
        st.metric("Critical RUL Threshold", f"{rel_settings.get('rul_threshold_critical', 0.05):.2%}",
                 help="RUL threshold for critical warnings (1%-10%)")
    with col_r2:
        st.metric("Gamma", f"{rel_settings.get('gamma', 0.99):.3f}",
                 help="Discount factor for future rewards (0.9-0.99)")
        st.metric("RUL Imbalance Penalty", f"{rel_settings.get('rul_imbalance_penalty', 15.0):.1f}",
                 help="Penalty for imbalanced RUL across PUs")
        st.metric("Warning RUL Threshold", f"{rel_settings.get('rul_threshold_warning', 0.10):.2%}",
                 help="RUL threshold for warnings (5%-20%)")
    with col_r3:
        st.metric("Failure Penalty", f"{rel_settings.get('failure_penalty', -2000.0):.0f}",
                 help="Penalty for PU failures (-5000 to -500)")
        st.metric("Critical RUL Penalty", f"{rel_settings.get('critical_rul_penalty', -800.0):.0f}",
                 help="Penalty for using PU with critical RUL")
        st.metric("Safe Operation Bonus", f"{rel_settings.get('safe_operation_bonus', 100.0):.0f}",
                 help="Bonus for keeping all PUs above safe threshold")
    
    st.info("💡 To modify these settings, chat with the Chief Engineer. Example: 'Increase performance weight to 0.6' or 'Tell Performance agent to increase power reward scale'.")


st.write('Copyright © 2024 Farraen. All rights reserved.')


#  ----------- Callbacks and updates ---------------

plot_results()
plot_iter()

# Check if bias changed and trigger re-optimization
if st.session_state.get('bias_changed', False) and st.session_state.get('rl_coordinator') is not None:
    if isinstance(st.session_state.df_1, pd.DataFrame) and len(st.session_state.df_1) > 0:
        df_temp = st.session_state.df_1.copy()
        tracks_to_optimize_idx = df_temp[df_temp["PU Actual"].isna()].index.tolist()
        if len(tracks_to_optimize_idx) > 0:
            # Re-optimize with new bias
            quick_optimisation(progress_bar=my_bar)
            status_placeholder.success('PU allocation updated with new bias', icon="✅")
    st.session_state.bias_changed = False

if start_button:
    # Always use quick optimization with pre-trained models
    quick_optimisation(progress_bar=my_bar)
    status_placeholder.success('PU allocation is successful', icon="✅")
    time.sleep(1)
    st.rerun()




 
