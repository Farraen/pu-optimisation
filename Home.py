# Developed by Farraen
# Date 2018
# Migrated to python 2023

import os, sys
import time
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

    # Load last result
    if isinstance(st.session_state.pu_results, dict) and len(st.session_state.pu_results) > 0:

        # Get all keys and sort them
        all_keys = sorted([k for k in st.session_state.pu_results.keys() if isinstance(k, (int, float))])
        
        if len(all_keys) == 0:
            return
        
        key = all_keys[-1]
        df_best = st.session_state.pu_results[key]

        fig = go.Figure()

        # Plot previous solutions (only if they exist)
        previous_keys = [k for k in all_keys if k < key]
        for idx, i in enumerate(previous_keys):
            if i in st.session_state.pu_results:
                flag = (idx == len(previous_keys) - 1)  # Show legend only for last previous solution
            dts = st.session_state.pu_results[i]
            fig.add_trace(go.Scatter(
                y=dts['RUL'],
                mode='lines',
                name='Previous solutions',
                showlegend = flag,
                line=dict(
                width=1,
                color='grey')
            ))

        # Plot best solution
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
            xanchor="right",  # changed
            x=1
        ))
        fig.update_layout(height=300,margin=dict(l=20, r=20, t=20, b=20))
        st.session_state.iterCompare_placeholder.plotly_chart(fig,use_container_width=True,height=300)


        fig2 = go.Figure()
        # Plot all available results
        for idx, i in enumerate(all_keys):
            if i not in st.session_state.pu_results:
                continue
            
            width = 1
            cc = 'grey'
            flag = True
            nameLegend='Previous solutions'
            flagRange = False
            if i == key:
                cc = 'blue'
                width = 4
                nameLegend='Best PU Allocation'
                flagRange = True
            elif idx == len(all_keys) - 2:  # Second to last
                flag = True
            else:
                flag = False

            dts = st.session_state.pu_results[i]

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
                        name=nameLegend,
                        showlegend = flag,
                        line=dict(
                        width=width,
                        color=cc)
                    ))
            
            if flagRange and len(maxpowerreduced) > 0:
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
            xanchor="right",  # changed
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
    """Quick optimization using pre-trained models - no retraining"""
    
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
    
    # Create RL environment
    env = PUSelectionEnv(
        track_data=df_temp,
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
    
    # Adjust reward weights based on current bias
    pu_bias = st.session_state.get('pu_bias', 2)
    if pu_bias == 1:  # High Performance
        coordinator.manager.update_reward_weights(
            performance_weight=0.6, reliability_weight=0.1
        )
    elif pu_bias == 10:  # Longer RUL
        coordinator.manager.update_reward_weights(
            performance_weight=0.1, reliability_weight=0.6
        )
    else:
        # Scale weights based on bias (1-10 scale)
        perf_weight = 0.1 + (10 - pu_bias) * 0.05
        rel_weight = 0.1 + (pu_bias - 1) * 0.05
        coordinator.manager.update_reward_weights(
            performance_weight=perf_weight, reliability_weight=rel_weight
        )
    
    st.session_state.rl_coordinator = coordinator
    
    # Use quick inference mode
    try:
        best_solution = coordinator.quick_inference(progress_bar=progress_bar)
        solution = make_full_solution(best_solution)
        
        # Calculate final metrics
        Fitness, PowerLoss, PowerLeft, RUL, PowerReduced = DamageModel(solution)
        
        st.session_state.df_1["PU Projection"] = solution
        st.session_state.df_1["PowerLeft"] = PowerLeft["PowerLeft"]
        st.session_state.df_1["PowerReduced"] = PowerReduced["PowerReduced"]
        st.session_state.df_1["RUL"] = RUL["RUL"]
        dts = st.session_state.df_1.copy()
        
        # Store final result
        st.session_state.pu_results[1] = dts  # Use simple key for quick results
        
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




# ----------- UI Section -------------------------
    
st_title('PU Decision Engine Playground')

st.caption("Optimized for dark mode. To change the theme, access the settings panel by clicking the three dots in the top-right corner of the app.")

with st.expander('Introduction',expanded=True):
    st_text('A virtual environment to demonstrate the ability of Hierarchical Multi-Agent Reinforcement Learning to solve PU selection problem. The system consists of a Manager agent that oversees rules and PU usage constraints, and two co-worker agents: Performance (minimizes degradation) and Reliability (ensures SOH > 0). Allows race engineer to quickly restrategise live with new race data and historical decisions. Adapted from Farraen\'s 2018 Matlab GA PU script and converted into Python environment with RL optimization. The UI was developed using 2018 season track data.')
    image = read_image("images/Page1_intro.png")
    st.image(image,width=700,use_column_width=True)

with st.expander('Damage model',expanded=False):
    st_text('The optimiser uses an artificial damage model made solely for demonstration purposes. The data does not represent true PU values.')
    image = read_image("images/Page1_damage.png")
    st.image(image,width=700,use_column_width=True)


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
        bias_value = st.select_slider(
            'Select bias:',
            options=['High Performance','2','3','4','5','6','7','8','9','Longer RUL'],
            value=('2'),
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
        st.session_state.num_episodes = st.number_input("Number of episodes (for training)", value=20, min_value=5, max_value=100, placeholder="Type a number...")
        
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




 
