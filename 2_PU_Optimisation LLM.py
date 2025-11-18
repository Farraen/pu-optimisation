# Developed by Farraen
# Date 2018
# Migrated to python 2023

import os, sys
import time
import streamlit as st
import pandas as pd
import numpy as np
import pygad
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


# ---------  System cache  -----------------

if 'pu_fitness_trace' not in st.session_state:
    st.session_state.pu_fitness_trace = []

if 'pu_results' not in st.session_state:
    st.session_state.pu_results = []

if 'df_2' not in st.session_state:
    st.session_state.df_2 = []

if 'bias' not in st.session_state:
    st.session_state.bias = 2

if 'gen_number' not in st.session_state:
    st.session_state.gen_number = []

if "iter" not in st.session_state:
    st.session_state.iter = []

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []
action = ""

if "prompt" not in st.session_state:
    st.session_state.prompt = []

if "text_input" not in st.session_state:
    st.session_state.text_input = []

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


# ---------- Prompt engine -------- 

dict_table_columns = {
  "Track":"Name of the track for the race", "PU Failures":"Column of PU failures where the user can manually assign in the event of a PU failure","PU Actual":"Column of real PU allocation after a race has ended",
  "PU Projection":"Prediction of PU allocation made by a decision engine. This is project early of the season. Engineers should rely on this prediction for each races",
  "MinTemp": "Minimum track temperature during the race day",
  "MaxTemp": "Maximum track temperature during the race day",
  "Distance": "Total race distance for an F1 car to complete the race",
  "PowerLeft":"The PU power left after PU degradation at the end of the season",
  "PowerReduced":"Total PU power reduction after at the end of the season",
  "RUL":"The remaining useful life of the PU in percentage. 100percent is like new PU and 0percent is a failed PU",
  "DamageThisRace":"Is the amount of PU degradation after a race"
  }
str_table_columns = json.dumps(dict_table_columns)



def send_message_secondary(persona,prompt):

    persona = [{"role":"system", "content":prompt}]
    user_messages = [{"role":"user", "content":prompt}]
    user_messages.extend(persona)
    openai_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": m["role"], "content": m["content"]} for m in user_messages])
    
    response_str = openai_response.choices[0].message.content

    return response_str

def check_question(prompt):

    index = []
    for i in range(0,3):
        str1 = f"You are an f1 racing engineer and has racing dashboard waiting for request from the user. "
        str8 = f"Based on the user prompt '{prompt}', is it a specific process task? Check if it is about initialising or populating PU allocation table or any tables, then it is a specific process task and answer yes. Check if it is about informing PU status or PU failures, then it is a specific process task and answer yes. Check if it is about reoptimising/restrategising/optimising PU allocation, then it is a specific process task and answer yes. Check if the user is informing about failed PU in a specific race or round, then it is a specific process task and answer yes."
        persona = [{"role":"system", "content":str1}]
        prompt = str8
        response_str = send_message_secondary(persona,prompt)
        flag = response_str.lower().startswith("yes")  
        index.append(flag)

    flag = statistics.median(index)

    return flag


def find_task(prompt):
    str1 = f"You are an f1 engineer and has racing dashboard waiting for request from the user. "

    dict_options = [
        "Initialise to fill in the PU projection/allocation table",
        "Assign or replace actual PU number for a race",
        "Restrategise/rerun/optimise PU projections/allocations",
        "failed PU"]
    str_options = json.dumps(dict_options)

    str8 = f"Based on the user prompt '{prompt}', find the closest item to the items in this list [{str_options}]. if there is no item in the list that is close to the prompt, the assign index '8'. if the user prompt is a specific technical process but not in the list [{str_options}], then assign index '4'. if the user prompt is about initialising table, then assign index '0'. if the user prompt is about informing failed PU, then assign index '3' . if the user asked about clearing the chat, then assign index '-1'. if the user prompt is about informing failed PU, then assign index '3' .Put the results in a dictionary with clostest index in 'index' key and closest item in 'value' key. Just output the dictionary in string format."

    persona = [{"role":"system", "content":str1}]
    prompt = str8
    index = []
    for i in range(0,5):
        response_str = send_message_secondary(persona,prompt)
        try:
            response_str = ast.literal_eval(re.search('({.+})', response_str).group(0))
            flag = response_str['index']
            index.append(int(flag))
        except:
            pass

    index = statistics.median(index)

    return index

def find_failed_pu(prompt):
    str5 = f"There are three power units that can be assigned for each race and they are identified as 1, 2 and 3."
    prompt = f"Based on the user prompt '{prompt}', which PU has failed? Do not include any explaination."

    persona = [{"role":"system", "content":str5}]
    response_str = send_message_secondary(persona,prompt)
    temp = re.findall(r'\d+', response_str)
    index = list(map(int, temp))

    prompt = f"Based on the user prompt '{prompt}', which race when the PU fail? Just answer the race index. Do not include any explaination."
    persona = [{"role":"system", "content":str5}]
    response_str = send_message_secondary(persona,prompt)
    temp = re.findall(r'\d+', response_str)
    race = list(map(int, temp))

    payload = [index,race]

    return payload

def check_recommendations(prompt):
    str1 = f"You are an f1 racing engineer and has racing dashboard waiting for request from the user. "
    prompt = f"Based on the user prompt '{prompt}', did the user ask for recommendations or advice? Answer yes or no."
    persona = [{"role":"system", "content":str1}]
    index = []
    for i in range(0,3):
        response_str = send_message_secondary(persona,prompt)
        flag = response_str.lower().startswith("yes")  
        index.append(flag)
    flag = statistics.median(index)
    return flag


def send_message_technical(prompt):

    df_2 = st.session_state.df_2
    a = df_2.columns.values.tolist()
    b = df_2.values.tolist()
    b.insert(0, a)
    ystr = "["
    for row in b:
        s = '[' + ', '.join(str(x) for x in row) + '],'
        ystr = ystr + s
    df_str = ystr + ']'

    str1 = f"You are an F1 race engineer and a data scientist. Your role would be analysing race telemetry and find patterns, trends and anomalies. PU or power unit is the engine of an F1 car."
    str4 = f"There is a table called PU allocation table of F1 power units. There are 21 rows for each races with one power unit allocated for each race. These are the columns and description in a dictionary string format: {str_table_columns}. "
    str5 = f"There are three power units that can be assigned for each race and they are identified as 1, 2 and 3."
    str6 = f"The PU are selected for each of the race depending on their performance and durability. The objective of the selection is the maximise the vehicle performance throughout the season while keeping all PU survive until the last race of the season."
    str7 = f"The power unit is the engine of the F1 car and it has range of RUL or remaining useful life between 0% to 100%. Power unit or PU with RUL above 0% indicates that the PU survived. The power of the PU is in terms of kW. The higher the kW, the better the PU performance. "
    
    str2 = f"This is a power unit allocation table in a list format with the first row is the column names: '{df_str}'. "
    str8 = f"If the user is asking about PU survival projection or prediction, the refer to just 'RUL' column and check they are all above 0%. If it is all above 0%, then report that all PU going to survive. Do show the whole column in the response."
    str9 = f"If the user is not asking about RUL, just answer the initial prompt."

    prompt = str2 + prompt + str8
    persona = [{"role":"system", "content":str1+str4+str5+str6+str7+str8}]

    user_messages = [{"role":"user", "content":prompt}]
    user_messages.extend(persona)
    openai_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": m["role"], "content": m["content"]} for m in user_messages],
        stream=True)
    
    return openai_response

def update_table(prompt):


    df_2 = st.session_state.df_2.copy()
    df_2 = df_2[['Track', 'PU Failures', 'PU Actual', 'PU Projection']]
    df_2['Round'] = df_2.index +1 
    df_2['Race number'] = df_2.index +1 


    a = df_2.columns.values.tolist()
    b = df_2.values.tolist()
    b.insert(0, a)
    ystr = "["
    for row in b:
        s = '[' + ', '.join(str(x) for x in row) + '],'
        ystr = ystr + s
    df_str = ystr + ']'

    str1 = f"You are an F1 race engineer and a data scientist. Your role would be analysing race telemetry and find patterns, trends and anomalies. PU or power unit is the engine of an F1 car."
    str4 = f"There is a table called PU allocation table of F1 power units. There are 21 rows for each races with one power unit allocated for each race. These are the columns and description in a dictionary string format: {str_table_columns}. "
    str2 = f"This is a power unit allocation table in a list format with the first row is the column names: '{df_str}'. "

    str3 = f"Based on the user prompt '{prompt}', determine the PU index number. Do not include any explanation."
    prompt = str2 + prompt + str3
    persona = [{"role":"system", "content":str1+str4}]
    response_str = send_message_secondary(persona,prompt)
    temp = re.findall(r'\d+', response_str)
    index = list(map(int, temp))

    str3 = f"Based on the user prompt '{prompt}', determine race number. if there is no race number and the user specify the race track name, then find the race track index based on the PU allocation table. Do not include any explanation."
    prompt = str2 + prompt + str3
    persona = [{"role":"system", "content":str1+str4}]
    response_str = send_message_secondary(persona,prompt)
    temp = re.findall(r'\d+', response_str)
    race = list(map(int, temp))

    payload = [index,race]

    return payload


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

    dtt = st.session_state.df_2 

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

    bias = (st.session_state.bias-1)/2
    fitness_value = PowerLoss - (bias)*np.sum(failedPU)


    return fitness_value, PowerLoss, PowerLeft, RUL, PowerReduced
        
def plot_results():

    # Load last result
    if isinstance(st.session_state.pu_results, dict):

        key = list(st.session_state.pu_results)[-1]
        df_best = st.session_state.pu_results[key]


        fig = go.Figure()

        for i in range(1,key):
            if i == key-1:
                flag = True
            else:
                flag = False

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
        for i in range(1,key+1):
            
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
            elif i == key-1:
                flag = True
            else:
                flag = False

            dts = st.session_state.pu_results[i]

            # Get list of PUs
            pu_list = dts["PU Projection"].unique().tolist()
            pu_list.sort()

            # Find all max reduced for each PU
            maxpowerreduced = []
            for k in pu_list:
                mx = np.max(dts.loc[np.where(dts["PU Projection"] == k)[0],'PowerReduced'].to_numpy())
                maxpowerreduced.append(mx)

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
            
            if flagRange:
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
    dta = pd.DataFrame(st.session_state.pu_fitness_trace,columns=["value"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=dta['value'],
        mode='markers+lines',
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig.update_yaxes(title_text='Fitness Value')
    fig.update_xaxes(title_text='Generation')
    fig.update_layout(height=300)

    st.session_state.pu_iter_placeholder.plotly_chart(fig,use_container_width=True,height=300)

def change_bias():
    bias = st.session_state.slider
    if bias == 'High Performance':
        st.session_state.bias = 1
    elif bias == 'Longer RUL':
        st.session_state.bias = 10
    else:
        st.session_state.bias = int(bias)

# --------  For optimisation  -------------

def make_full_solution(solution):
    if not st.session_state.df_2 ["PU Actual"].isna().all():
        dff = st.session_state.df_2.copy()
        tracks_left_idx = dff["PU Actual"].isna()
        actual = dff["PU Actual"].to_numpy()        
        solutionFull = dff["PU Projection"].to_numpy()
        solutionFull[~tracks_left_idx] =  actual[~tracks_left_idx]
        solutionFull[tracks_left_idx] =  solution
        solution = solutionFull
    return solution

def fitness_func(ga_instance, solution, solution_idx):
    
    solution = make_full_solution(solution)

    fitness_value, PowerLoss, PowerLeft, RUL, PowerReduced = DamageModel(solution)

    return fitness_value

def on_start(ga_instance):
    st.session_state.pu_fitness_trace = []
    st.session_state.pu_results = {data: [] for data in range(1,ga_instance.num_generations)}
        
def on_generation(ga_instance):

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    solution = make_full_solution(solution)

    Fitness, PowerLoss, PowerLeft, RUL, PowerReduced = DamageModel(solution)

    st.session_state.df_2 ["PU Projection"] = solution
    st.session_state.df_2 ["PowerLeft"] = PowerLeft["PowerLeft"]
    st.session_state.df_2 ["PowerReduced"] = PowerReduced["PowerReduced"]
    st.session_state.df_2 ["RUL"] = RUL["RUL"]
    dts = st.session_state.df_2.copy()

    index = ga_instance.generations_completed
    st.session_state.pu_results[index] = dts

    st.session_state.pu_fitness_trace.append(solution_fitness)
    dta = pd.DataFrame(st.session_state.pu_fitness_trace,columns=["value"])

    st.session_state.iter = st.session_state.iter + 1
    my_bar.progress(st.session_state.iter/st.session_state.gen_number,text='Restrategise PU allocation. Please wait...')


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=dta['value'],
        mode='markers+lines',
    ))
    fig.update_yaxes(title_text='Fitness Value')
    fig.update_xaxes(title_text='Generation')
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.session_state.pu_iter_placeholder.plotly_chart(fig,use_container_width=True)

def optimisation_sequence():

    fitness_function = fitness_func

    # Only optimise track with no actual PU results
    tracks_left = st.session_state.df_2 ["PU Actual"].isna().sum()

    num_parents_mating = 4
    sol_per_pop = 5
    num_genes = int(tracks_left)
    st.session_state.iter = 0

    # Check PUs left
    pu_available = [1,2,3]
    PU_failed = st.session_state.df_2 ["PU Failures"].dropna().unique().tolist()
    for item in PU_failed:
        if item in pu_available:
            pu_available.remove(item)
                
    if not pu_available:
        st.error("No PU left")
    else:
        ga_instance = pygad.GA(num_generations=st.session_state.gen_number,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        gene_space=pu_available,
                        gene_type=int,
                        on_start=on_start,
                        on_generation=on_generation,
                        mutation_type='random')
        
        ga_instance.run()

        # Latest results
        plot_results()
    
def reoptimise():
    
    df_2 = st.session_state.df_2.copy()
    mode_requested = []
    flag = False
    for index, updates in st.session_state["mastertable"].items():
        if "edited_rows" in index:
            for row, value in updates.items():
                for colname, cellvalue in value.items():
                    mode_requested.append(colname)
                    df_2.loc[row,colname] = cellvalue

                    # Only get update flag for actual manipulation not failure
                    # Flag is for rerunning the decision engine
                    if "PU Actual" in colname:
                        if df_2.loc[row,colname] == df_2.loc[row,"PU Projection"]:
                            flag = False
                        else:
                            flag = True


    st.session_state.df_2 = df_2

    if "PU Actual" in mode_requested:
        if flag:
            optimisation_sequence()
            status_placeholder.success('PU allocation is successful', icon="✅")
            time.sleep(0.5)

    if "PU Failures" in mode_requested:
        optimisation_sequence()
        status_placeholder.success('PU allocation is successful', icon="✅")
        time.sleep(0.5)


#---------- Load in track information -------------
   
if not isinstance(st.session_state.df_2 ,pd.DataFrame):
    
    # Initial condition
    df_2 = pd.read_excel('data/Page1_track.xlsx')
    
    st.session_state.df_2 = df_2.copy()
    df_2["PU Failures"] = np.nan
    df_2["PU Actual"] = np.nan
    df_2["PU Projection"] = np.nan
    Fitness, PowerLoss, PowerLeft, RUL, PowerReduced = DamageModel(df_2["PU Projection"].to_numpy())
    df_2["PowerLeft"] = PowerLeft["PowerLeft"]
    df_2["PowerReduced"] = PowerReduced["PowerReduced"]
    df_2["RUL"] = RUL["RUL"]


    df_2.insert(3, "PU Projection", df_2.pop("PU Projection"))
    df_2.insert(3, "PU Actual", df_2.pop("PU Actual"))
    df_2.insert(3, "PU Failures", df_2.pop("PU Failures"))
    df_2.pop('Date')
    df_2.pop('No')

    # Going to be the main table
    st.session_state.df_2 = df_2.copy()

else:

    # Simulate and repopulate the master table
    df_2 = st.session_state.df_2 
    Fitness, PowerLoss, PowerLeft, RUL, PowerReduced = DamageModel(df_2["PU Projection"].to_numpy())
    df_2["PowerLeft"] = PowerLeft["PowerLeft"]
    df_2["PowerReduced"] = PowerReduced["PowerReduced"]
    df_2["RUL"] = RUL["RUL"]
    
    # Going to be the main table
    st.session_state.df_2 = df_2

# ----------- prompt engine ----------------------
def send_message(messages):

    persona = [{"role":"system", "content":"You are a racing car engineer"}]
    persona.extend(messages)
    openai_response = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[{"role": m["role"], "content": m["content"]} for m in persona],
        stream=True)
    
    return openai_response



def chat_callback():
    prompt = st.session_state['10']
    st.session_state.prompt = prompt

def select_callback():
    prompt = st.session_state.select_input
    st.session_state.prompt = prompt

# ----------- UI Section -------------------------
    
st_title('PU Decision Engine + AI Race Engineer')

st.caption("Optimized for dark mode. To change the theme, access the settings panel by clicking the three dots in the top-right corner of the app.")

with st.expander('Introduction',expanded=True):
    st_text('A virtual environment to demonstrate the ability of Genetic Algorithm (an evolutionary algorithm) to solve PU selection problem. Allows race engineer to quickly restrategise live with new race data and historical decisions. Adapted from Farraen\'s 2018 Matlab GA PU script and converted into Python environment. Results may vary due to to the GA library behaviour. The UI was developed using 2018 season track data.')
    image = read_image("images/Page2_intro.png")
    st.image(image,width=700,use_column_width=True)

with st.expander('Damage model',expanded=False):
    st_text('The optimiser uses an artificial damage model made solely for demonstration purposes. The data does not represent true PU values.')
    image = read_image("images/Page1_damage.png")
    st.image(image,width=700,use_column_width=True)


with st.expander('PU selection optimisation',expanded=True):

    col1,col2 = st.columns([0.3,0.7],gap="medium")
    with col2:
        st.write('PU allocation strategy table (Auto-update)')
        st.markdown(
        """
        - :green[Initialise button]: Press to initialise the PU selection at start of the season. Use the GA settings panel to change the decision prioritisation to either performance or durability.
        - :blue[Actual column]: Use the column with actual selection for completed races and GA will decide the best allocation for the next races. If there is no change, then GA will not trigger.
        - :red[Failure column]: Use the Failure column to exclude failed PU (fill in using PU index 1,2,3,..)
        """
        )

        start_button = st.button('Initialise')
        my_bar = st.progress(0)

        # Highligh rows depending on type (actual or projection)
        df_track_styled = st.session_state.df_2.copy()

        actual_row = np.where(~df_track_styled["PU Actual"].isna())[0]
        projection_row = np.where(df_track_styled["PU Actual"].isna())[0]

        df_track_styled = df_track_styled.style.set_properties(subset = pd.IndexSlice[actual_row, :], **{'background-color' : 'darkgreen'})\
        .set_properties(subset = pd.IndexSlice[projection_row, :], **{'background-color' : 'midnightblue'})

        # create data editor for the master table
        st.data_editor(df_track_styled,use_container_width=True,key='mastertable',disabled=["PU Projection"],on_change=reoptimise)
        status_placeholder = st.empty()


    with col1:
        
        st.write('AI race engineer')
     
        st.selectbox('You might want to try these prompts...',
            ["How to select a power unit for an F1 car?",
                "What is the best way to select power unit for each race? what factors are taken into account?",
                "Initialise pu allocation table",
                "PU number 1 has failed at race 20",
                "PU 1 has failed at race 3, make new recommendations.",
                "Restrategise",
                "Actual PU for race 2 is 1",
                "Actual PU for monaco race is 3",
                "PU number 3 failed at race 2",
                "How to optimise a racing line?",
                "When is the first f1 race?",
                "Which F1 tracks are power lap tracks?",
                "Which track is the hottest?"],index=None,key="select_input",on_change=select_callback)

        st.text_input("Enter prompt here:",key=10,on_change=chat_callback)

        for message in st.session_state.messages:
            if "user" in message["role"]:
                st.write(f"🧑    {message['content']}")
            else:
                st.write(f"🤖    :blue[{message['content']}]")

        
        if st.session_state.prompt:
            prompt = st.session_state.prompt
            # User
            st.write(f"🧑   {prompt}")

            # Assistant
            st.session_state.messages.append({"role": "user", "content": prompt})

            index = int(find_task(prompt))
            print()

            # Initialise
            if index == -1:
                # Done!
                action = 'clear chat'

            elif index == 0:
                # Done!
                action = 'initialise table'
            
            elif index == 1:
                action = 'update table'
                payload = update_table(prompt)
            
            elif index == 2:
                # Done!
                action = 'initialise table'   
            
            elif index == 3:
                action = 'failed pu'
                payload = find_failed_pu(prompt)
                payload2 = check_recommendations(prompt)

            
            elif index == 4:
                openai_response = send_message_technical(prompt)
                action = 'technical'

                message_placeholder = st.empty()
                full_response = ""
                for response in openai_response:
                    full_response += (response.choices[0].delta.content or "")
                    message_placeholder.write(f"🤖    :blue[{full_response}]▌")
                message_placeholder.write(f"🤖    :blue[{full_response}]")
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            else:
                openai_response = send_message(st.session_state.messages)
                
                message_placeholder = st.empty()
                full_response = ""
                for response in openai_response:
                    full_response += (response.choices[0].delta.content or "")
                    message_placeholder.write(f"🤖    :blue[{full_response}]▌")
                message_placeholder.write(f"🤖    :blue[{full_response}]")

                st.session_state.messages.append({"role": "assistant", "content": full_response})

            st.session_state.messages = st.session_state.messages[-6:]

with st.expander('Optimisation results',expanded=True):
    col1,col2 = st.columns([1,1])
    with col1:
        st.session_state.iterCompare_placeholder = st.empty()
    
    with col2:
        st.session_state.PUCompare_placeholder = st.empty()


with st.expander('Genetic algorithm settings',expanded=True):

    col1, col2 = st.columns([0.8,1],gap='Small')

    with col1:

        st.select_slider(
            'Select bias:',
            options=['High Performance','2','3','4','5','6','7','8','9','Longer RUL'],
            value=('2'),
            on_change=change_bias,
            key='slider')    
           
        st.session_state.gen_number = st.number_input("Number of generation", value=20, placeholder="Type a number...")

    with col2:
        st.session_state.pu_iter_placeholder = st.empty()

    


st.write('Copyright © 2024 Farraen. All rights reserved.')


#  ----------- Callbacks and updates ---------------

plot_results()
plot_iter()

st.session_state.prompt = []


if 'initialise table' in action:
    optimisation_sequence()
    status_placeholder.success('PU allocation is successful', icon="✅")
    time.sleep(1)

    st.session_state.messages.append({"role": "assistant", "content": 'Done.'})
    st.rerun()    


if start_button:
    optimisation_sequence()
    status_placeholder.success('PU allocation is successful', icon="✅")
    time.sleep(1)
    st.rerun()

if 'failed pu' in action:
    
    # Update table
    df_2 = st.session_state.df_2.copy()
    PU_failed = payload[0]
    Race_affected = payload[1]
    df_2.loc[Race_affected[0]-1,'PU Failures'] = PU_failed[0]
    st.session_state.df_2 = df_2
    
    # Rerun optimisation sequence
    optimisation_sequence()
    status_placeholder.success('PU allocation updated.', icon="✅")
    time.sleep(1)

    # Check if AI make recommedations
    if not payload2:
        st.session_state.messages.append({"role": "assistant", "content": 'Done.'})
    else:
        st.session_state.messages.append({"role": "assistant", "content": payload2[0]})
  
    st.rerun()    

if 'update table' in action:
    # Update table
    df_2 = st.session_state.df_2.copy()
    Actual = payload[0]
    Race_affected = payload[1]
    df_2.loc[Race_affected[0]-1,'PU Actual'] = Actual[0]
    st.session_state.df_2 = df_2
    
    # Rerun optimisation sequence
    optimisation_sequence()
    status_placeholder.success('PU allocation updated.', icon="✅")
    time.sleep(1)

    # Check if AI make recommedations
    st.session_state.messages.append({"role": "assistant", "content": 'Done.'})
          
    st.rerun()    

if 'clear chat' in action:

    st.session_state.messages = []
    st.rerun()    




 
