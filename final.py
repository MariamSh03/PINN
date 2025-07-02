import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
import io
import pickle
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import threading

# Configure page
st.set_page_config(
    page_title="PINN ODE Solver",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for clean light interface
# st.markdown("""
# <style>
#     /* FORCE LIGHT THEME EVERYWHERE */
#     .stApp {
#         background-color: #ffffff !important;
#         color: #000000 !important;
#     }
    
#     /* SIDEBAR - LIGHT BACKGROUND */
#     .css-1d391kg, .css-1lcbmhc, .css-17eq0hr, section[data-testid="stSidebar"] {
#         background-color: #f8f9fa !important;
#     }
    
#     .css-1d391kg > div, .css-1lcbmhc > div {
#         background-color: #f8f9fa !important;
#     }
    
#     /* MAIN CONTENT AREA - WHITE */
#     .main .block-container {
#         background-color: #ffffff !important;
#         padding-top: 2rem;
#     }
    
#     /* NAVIGATION HEADER - LIGHT */
#     header[data-testid="stHeader"] {
#         background-color: #f8f9fa !important;
#     }
    
#     .css-18e3th9, .css-1d391kg, .css-k1vhr4, header {
#         background-color: #f8f9fa !important;
#     }
    
#     /* ALL TEXT DARK */
#     h1, h2, h3, h4, h5, h6 {
#         color: #000000 !important;
#     }
    
#     .stMarkdown, .stMarkdown p, .stText, div[data-testid="stMarkdownContainer"] {
#         color: #000000 !important;
#     }
    
#     /* SIDEBAR TEXT DARK */
#     .css-1d391kg .stMarkdown, .css-1d391kg .stText, .css-1d391kg label {
#         color: #000000 !important;
#     }
    
#          /* SIDEBAR ELEMENTS - LIGHT WITH DARK TEXT */
#      .stSelectbox > div > div {
#          background-color: #ffffff !important;
#          color: #000000 !important;
#          border: 1px solid #cccccc !important;
#      }
     
#      .stSelectbox label {
#          color: #000000 !important;
#      }
     
#      /* DROPDOWN TEXT AND OPTIONS */
#      .stSelectbox > div > div > div {
#          color: #000000 !important;
#          background-color: #ffffff !important;
#      }
     
#      .stSelectbox option {
#          color: #000000 !important;
#          background-color: #ffffff !important;
#      }
     
#     /* SLIDER INPUT FIXES */
#     input[type="range"] {
#         background: #ffffff !important;
#         color: #000000 !important;
#     }

#     /* SLIDER TICKS AND TEXT */
#     .stSlider div[data-testid="stTickBar"] span {
#         color: #000000 !important;
#     }

#     .stSlider label, .stSlider span, .stSlider div {
#         color: #000000 !important;
#     }

#     /* ACCESSIBILITY-FRIENDLY FALLBACKS */
#     div[role="slider"], div[aria-label*="slider"] {
#         color: #000000 !important;
#         background-color: #ffffff !important;
#     }

#     /* DARK TEXT FOR ANY REMAINING COMPONENTS */
#     [data-testid*="label"], [data-testid*="stSlider"], [data-testid*="stNumberInput"] {
#         color: #000000 !important;
#     }

#         /* NUMBER INPUTS */
#     .stNumberInput > div > div > input {
#         background-color: #ffffff !important;
#         color: #000000 !important;
#         border: 1px solid #cccccc !important;
#     }
    
#     .stNumberInput label {
#         color: #000000 !important;
#     }
    
#     /* BUTTONS */
#     .stButton > button {
#         background-color: #007bff !important;
#         color: white !important;
#         border: none !important;
#         border-radius: 5px;
#         transition: all 0.3s;
#     }
    
#     .stButton > button:hover {
#         background-color: #0056b3 !important;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
    
#     .stButton > button[kind="primary"] {
#         background-color: #28a745 !important;
#         color: white !important;
#     }
    
#     .stButton > button[kind="primary"]:hover {
#         background-color: #1e7e34 !important;
#     }
    
#     /* PLOT CONTAINERS - WHITE BACKGROUND */
#     .stPlotlyChart {
#         border: 1px solid #e0e0e0 !important;
#         border-radius: 8px;
#         padding: 15px;
#         margin: 10px 0;
#         background-color: #ffffff !important;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
    
#     /* FORCE PLOTLY PLOT BACKGROUNDS WHITE */
#     .js-plotly-plot .plotly .modebar {
#         background-color: #ffffff !important;
#     }
    
#     .js-plotly-plot .plotly .main-svg {
#         background-color: #ffffff !important;
#     }
    
#     /* METRIC CARDS - LIGHT */
#     .metric-card {
#         background-color: #f8f9fa !important;
#         border: 1px solid #e9ecef;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 0.5rem 0;
#         box-shadow: 0 1px 3px rgba(0,0,0,0.1);
#     }
    
#     [data-testid="metric-container"] {
#         background-color: #f8f9fa !important;
#         border: 1px solid #e9ecef !important;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         box-shadow: 0 1px 3px rgba(0,0,0,0.1);
#         color: #000000 !important;
#     }
    
#     [data-testid="metric-container"] > div {
#         color: #000000 !important;
#     }
    
#     /* PROGRESS BAR */
#     .stProgress .st-bo {
#         background-color: #007bff !important;
#     }
    
#     .stProgress > div > div > div {
#         background-color: #e9ecef !important;
#     }
    
#     /* EXPANDERS - LIGHT */
#     .streamlit-expanderHeader {
#         background-color: #f8f9fa !important;
#         border: 1px solid #e9ecef !important;
#         border-radius: 5px;
#         color: #000000 !important;
#     }
    
#     .streamlit-expanderContent {
#         background-color: #ffffff !important;
#         border: 1px solid #e9ecef !important;
#         border-top: none;
#         color: #000000 !important;
#     }
    
#     /* TABS - LIGHT */
#     .stTabs [data-baseweb="tab-list"] {
#         background-color: #f8f9fa !important;
#         border-bottom: 1px solid #e9ecef !important;
#     }
    
#     .stTabs [data-baseweb="tab"] {
#         color: #000000 !important;
#         background-color: transparent !important;
#     }
    
#     .stTabs [aria-selected="true"] {
#         background-color: #007bff !important;
#         color: white !important;
#     }
    
#     /* MULTISELECT */
#     .stMultiSelect > div {
#         background-color: #ffffff !important;
#     }
    
#     .stMultiSelect label {
#         color: #000000 !important;
#     }
    
#     /* CODE BLOCKS */
#     .stCodeBlock {
#         background-color: #f8f9fa !important;
#         border: 1px solid #e9ecef !important;
#         color: #000000 !important;
#     }
    
#     /* TABLES */
#     .dataframe {
#         background-color: #ffffff !important;
#         color: #000000 !important;
#     }
    
#     .dataframe th {
#         background-color: #f8f9fa !important;
#         color: #000000 !important;
#     }
    
#     /* SUCCESS/ERROR MESSAGES */
#     .stSuccess {
#         background-color: #d4edda !important;
#         border-color: #c3e6cb !important;
#         color: #155724 !important;
#     }
    
#     .stError {
#         background-color: #f8d7da !important;
#         border-color: #f5c6cb !important;
#         color: #721c24 !important;
#     }
    
#     .stWarning {
#         background-color: #fff3cd !important;
#         border-color: #ffeaa7 !important;
#         color: #856404 !important;
#     }
    
#     /* CHECKBOX */
#     .stCheckbox label {
#         color: #000000 !important;
#     }
    
#     /* RADIO */
#     .stRadio label {
#         color: #000000 !important;
#     }
    
#          /* ENSURE ALL LABELS ARE DARK */
#      label {
#          color: #000000 !important;
#      }
     
#      /* FORCE ALL INPUT TEXT DARK */
#      input {
#          color: #000000 !important;
#      }
     
#      /* FORCE ALL SPANS AND DIVS IN SIDEBAR DARK */
#      .css-1d391kg span, .css-1d391kg div {
#          color: #000000 !important;
#      }
     
#      /* ADDITIONAL TEXT ELEMENTS */
#      .stMarkdown span, .stText span {
#          color: #000000 !important;
#      }
     
#      /* FORCE WHITE BACKGROUNDS ON ALL MAIN CONTAINERS */
#      div[data-testid="stVerticalBlock"] {
#          background-color: #ffffff !important;
#      }
     
#      div[data-testid="stHorizontalBlock"] {
#          background-color: #ffffff !important;
#      }
     
#      /* OVERRIDE ANY REMAINING COLORED TEXT */
#      .st-emotion-cache-1y4p8pa, .st-emotion-cache-16idsys {
#          color: #000000 !important;
#      }
    
#     /* Custom styling for scientific notation */
#     .metric-value {
#         font-family: 'Courier New', monospace;
#         font-weight: bold;
#         color: #000000 !important;
#     }
# </style>
# """, unsafe_allow_html=True)

import tensorflow as tf
import numpy as np
from typing import List

class ODEParameters:
    """Container for ODE system parameters"""
    def __init__(self):
        # ODE System Parameters
        self.r = 0.01      # Tumor growth rate (1/day)
        self.K = 1e6       # Carrying capacity (cells)
        self.f = 0.01      # Dystrophin impact (1/day)
        self.k = 0.01      # Dystrophin degradation (1/day)
        self.g = 0.01      # Dystrophin activation (1/day)
        self.h = 0.01      # Recovery rate (1/day)
        self.S0 = 2.0      # Reference staging (units)
        self.j = 0.01      # Feedback strength (1/day)
        
        # Initial conditions
        self.T0 = 1000.0   # Initial tumor size (cells)
        self.D0 = 10.0     # Initial dystrophin (units)
        self.S0_ic = 3.0   # Initial staging impact (units)
        
        # Time domain
        self.t_start = 0.0
        self.t_end = 100.0
        self.n_points = 1000

    def to_dict(self):
        return {
            'r': self.r, 'K': self.K, 'f': self.f, 'k': self.k,
            'g': self.g, 'h': self.h, 'S0': self.S0, 'j': self.j,
            'T0': self.T0, 'D0': self.D0, 'S0_ic': self.S0_ic,
            't_start': self.t_start, 't_end': self.t_end, 'n_points': self.n_points
        }
    
    def from_dict(self, params_dict):
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

class PINNModel(tf.keras.Model):
    """Neural network model for PINN"""
    def __init__(self, layers: List[int], activation: str = 'tanh'):
        super(PINNModel, self).__init__()
        self.layers_list = []
        
        # Build hidden layers with proper initialization
        for i in range(len(layers) - 2):
            self.layers_list.append(
                tf.keras.layers.Dense(
                    layers[i+1], 
                    activation=activation,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                    bias_initializer='zeros'
                )
            )
        
        # Output layer (3 outputs: T, D, S) - no activation for final layer
        self.layers_list.append(
            tf.keras.layers.Dense(
                3, 
                activation=None,
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                bias_initializer='zeros'
            )
        )
    
    def call(self, t):
        # Normalize input time to [0,1] for better numerical stability
        t_norm = t / 100.0  # assuming max time is 100
        x = t_norm
        for layer in self.layers_list:
            x = layer(x)
        return x

def runge_kutta_4th_order(params: ODEParameters):
    """Solve the ODE system using 4th order Runge-Kutta method"""
    t_span = np.linspace(params.t_start, params.t_end, params.n_points)
    dt = (params.t_end - params.t_start) / (params.n_points - 1)
    
    # Initialize solution arrays
    T = np.zeros(params.n_points)
    D = np.zeros(params.n_points)
    S = np.zeros(params.n_points)
    
    # Initial conditions
    T[0] = params.T0
    D[0] = params.D0
    S[0] = params.S0_ic
    
    def system_ode(t, y):
        """ODE system: y = [T, D, S]"""
        T_val, D_val, S_val = y
        
        # Ensure positive values
        T_val = max(T_val, 1e-10)
        D_val = max(D_val, 1e-10)
        S_val = max(S_val, 1e-10)
        
        dT_dt = params.r * T_val * (1 - T_val/params.K) - params.f * D_val * T_val
        dD_dt = -params.k * D_val + params.g * S_val * D_val
        dS_dt = params.h * (params.S0 - S_val) - params.j * D_val * S_val
        
        return np.array([dT_dt, dD_dt, dS_dt])
    
    # RK4 integration
    for i in range(params.n_points - 1):
        y = np.array([T[i], D[i], S[i]])
        t = t_span[i]
        
        k1 = dt * system_ode(t, y)
        k2 = dt * system_ode(t + dt/2, y + k1/2)
        k3 = dt * system_ode(t + dt/2, y + k2/2)
        k4 = dt * system_ode(t + dt, y + k3)
        
        y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        T[i+1] = y_next[0]
        D[i+1] = y_next[1]
        S[i+1] = y_next[2]
    
    return t_span, np.column_stack([T, D, S])

class PINNSolver:
    """PINN solver for the tumor-dystrophin-staging system"""
    
    def __init__(self, params: ODEParameters, layers: List[int],
                 activation: str = 'tanh', learning_rate: float = 0.001):
        self.params = params
        self.model = PINNModel(layers, activation)
        
        # Use adaptive learning rate with clipping
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0  # Gradient clipping to prevent explosion
        )
        
        # Training data with proper scaling
        t_raw = tf.linspace(params.t_start, params.t_end, params.n_points)
        self.t_train = tf.reshape(tf.cast(t_raw, tf.float32), (-1, 1))
        self.t_ic = tf.constant([[params.t_start]], dtype=tf.float32)
        
        # Loss history
        self.loss_history = {'total': [], 'physics': [], 'ic': []}
        self.training_active = False
        
        # Scaling factors for numerical stability
        self.T_scale = 1000.0  # Scale T to order of 1
        self.D_scale = 10.0    # Scale D to order of 1  
        self.S_scale = 1.0     # S already order of 1
    
    def physics_loss(self, t):
        """Compute physics-informed loss with numerical stability"""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            u_raw = self.model(t)
            
            # Scale outputs to proper physical ranges
            T = u_raw[:, 0:1] * self.T_scale
            D = u_raw[:, 1:2] * self.D_scale  
            S = u_raw[:, 2:3] * self.S_scale
            
        # Compute gradients
        dT_dt = tape.gradient(T, t)
        dD_dt = tape.gradient(D, t) 
        dS_dt = tape.gradient(S, t)
        
        del tape
        
        # Check for None gradients
        if dT_dt is None or dD_dt is None or dS_dt is None:
            return tf.constant(1e6, dtype=tf.float32)
        
        # ODE residuals with safe operations
        T_safe = tf.clip_by_value(T, 1e-6, self.params.K)
        D_safe = tf.clip_by_value(D, 1e-6, 1e6)
        S_safe = tf.clip_by_value(S, 1e-6, 1e6)
        
        # Residuals
        r1 = dT_dt - (self.params.r * T_safe * (1 - T_safe/self.params.K) - self.params.f * D_safe * T_safe)
        r2 = dD_dt - (-self.params.k * D_safe + self.params.g * S_safe * D_safe)
        r3 = dS_dt - (self.params.h * (self.params.S0 - S_safe) - self.params.j * D_safe * S_safe)
        
        # Compute losses with numerical stability
        loss_1 = tf.reduce_mean(tf.square(r1)) / (self.T_scale**2)
        loss_2 = tf.reduce_mean(tf.square(r2)) / (self.D_scale**2)
        loss_3 = tf.reduce_mean(tf.square(r3)) / (self.S_scale**2)
        
        total_physics_loss = loss_1 + loss_2 + loss_3
        
        # Check for NaN and return large value if found
        return tf.cond(
            tf.math.is_nan(total_physics_loss),
            lambda: tf.constant(1e6, dtype=tf.float32),
            lambda: total_physics_loss
        )
    
    def initial_condition_loss(self):
        """Compute initial condition loss with scaling"""
        u_ic_raw = self.model(self.t_ic)
        
        # Scale outputs
        T_ic = u_ic_raw[:, 0:1] * self.T_scale
        D_ic = u_ic_raw[:, 1:2] * self.D_scale
        S_ic = u_ic_raw[:, 2:3] * self.S_scale
        
        # Normalized losses
        loss_T = tf.reduce_mean(tf.square((T_ic - self.params.T0) / self.T_scale))
        loss_D = tf.reduce_mean(tf.square((D_ic - self.params.D0) / self.D_scale))
        loss_S = tf.reduce_mean(tf.square((S_ic - self.params.S0_ic) / self.S_scale))
        
        total_ic_loss = loss_T + loss_D + loss_S
        
        # Check for NaN
        return tf.cond(
            tf.math.is_nan(total_ic_loss),
            lambda: tf.constant(1e6, dtype=tf.float32),
            lambda: total_ic_loss
        )
    
    @tf.function
    def train_step(self, ic_weight=1.0):
        """Single training step with gradient clipping"""
        with tf.GradientTape() as tape:
            physics_loss_val = self.physics_loss(self.t_train)
            ic_loss_val = self.initial_condition_loss()
            total_loss = physics_loss_val + ic_weight * ic_loss_val
        
        # Compute gradients with checks
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Check for None gradients
        if any(g is None for g in gradients):
            return tf.constant(1e6), physics_loss_val, ic_loss_val
        
        # Apply gradients with clipping
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss, physics_loss_val, ic_loss_val
    
    def predict(self, t_eval):
        """Generate predictions with proper scaling"""
        if isinstance(t_eval, (list, np.ndarray)):
            t_eval = np.array(t_eval, dtype=np.float32)
        
        t_eval_tf = tf.reshape(tf.constant(t_eval, dtype=tf.float32), (-1, 1))
        u_raw = self.model(t_eval_tf).numpy()
        
        # Scale back to physical units
        T = u_raw[:, 0] * self.T_scale
        D = u_raw[:, 1] * self.D_scale
        S = u_raw[:, 2] * self.S_scale
        
        return np.column_stack([T, D, S])
    
# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'params' not in st.session_state:
        st.session_state.params = ODEParameters()
    if 'solver' not in st.session_state:
        st.session_state.solver = None
    if 'training_active' not in st.session_state:
        st.session_state.training_active = False
    if 'training_thread' not in st.session_state:
        st.session_state.training_thread = None
    if 'current_epoch' not in st.session_state:
        st.session_state.current_epoch = 0
    if 'total_epochs' not in st.session_state:
        st.session_state.total_epochs = 5000
    if 'plot_layout' not in st.session_state:
        st.session_state.plot_layout = 'vertical'
    if 'show_combined' not in st.session_state:
        st.session_state.show_combined = False


# Utility functions
def create_model_directory():
    """Create models directory if it doesn't exist"""
    if not os.path.exists('models'):
        os.makedirs('models')

def save_model(solver, name, params):
    """Save trained model and parameters"""
    create_model_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"models/{name}_{timestamp}"
    
    # Save model weights (using .weights.h5 extension for newer TensorFlow versions)
    solver.model.save_weights(f"{filename}.weights.h5")
    
    # Save parameters and training history
    model_data = {
        'params': params.to_dict(),
        'loss_history': solver.loss_history,
        'architecture': {
            'layers': [layer.units for layer in solver.model.hidden_layers],
            'activation': 'tanh'  # Default assumption
        }
    }
    
    with open(f"{filename}_data.json", 'w') as f:
        json.dump(model_data, f, indent=2)
    
    return filename

def load_saved_models():
    """Load list of saved models"""
    if not os.path.exists('models'):
        return []
    
    models = []
    for file in os.listdir('models'):
        if file.endswith('_data.json'):
            models.append(file.replace('_data.json', ''))
    return sorted(models, reverse=True)

def load_parameter_presets():
    """Load parameter presets from JSON file"""
    presets = {
        "default_parameters": {
            "name": "Default Configuration",
            "parameters": {
                "r": 0.01, "K": 1000000, "f": 0.01, "k": 0.01,
                "g": 0.01, "h": 0.01, "S0": 2.0, "j": 0.01,
                "T0": 1000.0, "D0": 10.0, "S0_ic": 3.0,
                "t_start": 0.0, "t_end": 100.0, "n_points": 1000
            }
        },
        "aggressive_growth": {
            "name": "Aggressive Tumor Growth",
            "parameters": {
                "r": 0.05, "K": 2000000, "f": 0.005, "k": 0.02,
                "g": 0.005, "h": 0.005, "S0": 1.5, "j": 0.015,
                "T0": 2000.0, "D0": 5.0, "S0_ic": 4.0,
                "t_start": 0.0, "t_end": 50.0, "n_points": 1000
            }
        },
        "therapeutic_intervention": {
            "name": "Therapeutic Intervention",
            "parameters": {
                "r": 0.008, "K": 800000, "f": 0.02, "k": 0.005,
                "g": 0.02, "h": 0.025, "S0": 1.0, "j": 0.008,
                "T0": 5000.0, "D0": 20.0, "S0_ic": 2.0,
                "t_start": 0.0, "t_end": 150.0, "n_points": 1500
            }
        }
    }
    return presets

# UI Components

def render_sidebar():
    """Render the sidebar with parameter controls"""
    st.sidebar.title("🎛️ Control Panel")
    
    # Parameter Presets
    st.sidebar.subheader("📋 Parameter Presets")
    presets = load_parameter_presets()
    preset_names = [presets[key]["name"] for key in presets.keys()]
    selected_preset = st.sidebar.selectbox("Load Preset", ["Custom"] + preset_names)
    
    if selected_preset != "Custom":
        # Find the preset key
        preset_key = None
        for key, preset in presets.items():
            if preset["name"] == selected_preset:
                preset_key = key
                break
        
        if preset_key and st.sidebar.button("Apply Preset"):
            st.session_state.params.from_dict(presets[preset_key]["parameters"])
            st.success(f"Applied preset: {selected_preset}")
            st.rerun()
    
    # ODE System Parameters
    st.sidebar.subheader("📊 ODE System Parameters")
    
    params = st.session_state.params
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        params.r = st.slider("r (Growth rate)", 0.001, 0.1, params.r, 0.001, format="%.3f")
        params.f = st.slider("f (Dystrophin impact)", 0.001, 0.1, params.f, 0.001, format="%.3f")
        params.k = st.slider("k (Degradation)", 0.001, 0.2, params.k, 0.001, format="%.3f")
        params.g = st.slider("g (Activation)", 0.001, 0.1, params.g, 0.001, format="%.3f")
    
    with col2:
        params.h = st.slider("h (Recovery rate)", 0.001, 0.2, params.h, 0.001, format="%.3f")
        params.j = st.slider("j (Feedback)", 0.001, 0.1, params.j, 0.001, format="%.3f")
        params.S0 = st.slider("S₀ (Reference)", 0.1, 10.0, params.S0, 0.1, format="%.1f")
    
    params.K = st.sidebar.number_input("K (Carrying capacity)", 
                                      min_value=1000.0, max_value=1e8, 
                                      value=float(params.K), format="%.0e")
    
    # Initial Conditions
    st.sidebar.subheader("🎯 Initial Conditions")
    params.T0 = st.sidebar.number_input("T(0) - Initial tumor size", 
                                       min_value=1.0, max_value=100000.0, 
                                       value=params.T0)
    params.D0 = st.sidebar.number_input("D(0) - Initial dystrophin", 
                                       min_value=0.1, max_value=100.0, 
                                       value=params.D0)
    params.S0_ic = st.sidebar.number_input("S(0) - Initial staging", 
                                          min_value=0.1, max_value=20.0, 
                                          value=params.S0_ic)
    
    # Time Domain Settings
    st.sidebar.subheader("⏰ Time Domain")
    params.t_start = st.sidebar.number_input("Start time", 0.0, 10.0, params.t_start)
    params.t_end = st.sidebar.slider("End time", 10, 500, int(params.t_end))
    params.n_points = st.sidebar.slider("Evaluation points", 100, 2000, params.n_points)
    
    # PINN Architecture
    st.sidebar.subheader("🧠 PINN Architecture")
    
    # Number of hidden layers
    num_layers = st.sidebar.slider("Number of hidden layers", 1, 5, 3)
    
    # Individual layer sizes
    selected_layers = []
    for i in range(num_layers):
        layer_size = st.sidebar.selectbox(
            f"Layer {i+1} size", 
            [20, 50, 100, 200], 
            index=1,  # Default to 50
            key=f"layer_{i}"
        )
        selected_layers.append(layer_size)
    
    activation = st.sidebar.selectbox("Activation function", 
                                     ['tanh', 'relu', 'sigmoid'], 
                                     index=0)
    
    learning_rate = st.sidebar.select_slider("Learning rate",
                                           options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                                           value=0.001,
                                           format_func=lambda x: f"{x:.4f}")
    
    epochs = st.sidebar.slider("Training epochs", 1000, 20000, 5000, 500)
    ic_weight = st.sidebar.slider("IC weight", 1.0, 100.0, 10.0, 1.0)
    
    # Model Management
    st.sidebar.subheader("💾 Model Management")
    saved_models = load_saved_models()
    if saved_models:
        selected_model = st.sidebar.selectbox("Saved Models", ["None"] + saved_models)
        if selected_model != "None" and st.sidebar.button("Load Model"):
            try:
                # Load model data
                with open(f"models/{selected_model}_data.json", 'r') as f:
                    model_data = json.load(f)
                
                # Update parameters
                st.session_state.params.from_dict(model_data['params'])
                
                # Create new solver with loaded architecture
                arch = model_data['architecture']
                st.session_state.solver = PINNSolver(
                    st.session_state.params, 
                    arch['layers'], 
                    arch['activation']
                )
                
                # Load weights
                st.session_state.solver.model.load_weights(f"models/{selected_model}.weights.h5")
                st.session_state.solver.loss_history = model_data['loss_history']
                
                st.success(f"Loaded model: {selected_model}")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
    
    return selected_layers, activation, learning_rate, epochs, ic_weight

def render_training_controls():
    """Render training control buttons"""
    st.subheader("🚀 Training Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        start_training = st.button("▶️ Start Training", 
                                  disabled=st.session_state.training_active,
                                  type="primary")
    
    with col2:
        stop_training = st.button("⏹️ Stop Training", 
                                 disabled=not st.session_state.training_active)
    
    with col3:
        reset_params = st.button("🔄 Reset Parameters")
    
    with col4:
        save_current = st.button("💾 Save Model")
    
    return start_training, stop_training, reset_params, save_current

def create_loss_plots(solver):
    """Create real-time loss plots"""
    if not solver or not solver.loss_history['total']:
        return None
    
    history = solver.loss_history
    epochs = list(range(len(history['total'])))
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Loss', 'Physics Loss', 'IC Loss', 'Combined View'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Total loss
    fig.add_trace(
        go.Scatter(x=epochs, y=history['total'], name='Total Loss', line=dict(color='red')),
        row=1, col=1
    )
    
    # Physics loss
    fig.add_trace(
        go.Scatter(x=epochs, y=history['physics'], name='Physics Loss', line=dict(color='blue')),
        row=1, col=2
    )
    
    # IC loss
    fig.add_trace(
        go.Scatter(x=epochs, y=history['ic'], name='IC Loss', line=dict(color='green')),
        row=2, col=1
    )
    
    # Combined view
    fig.add_trace(
        go.Scatter(x=epochs, y=history['total'], name='Total', line=dict(color='red')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['physics'], name='Physics', line=dict(color='blue')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['ic'], name='IC', line=dict(color='green')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True,
                     plot_bgcolor='rgba(0,0,0,0)',
                     paper_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss", type="log")
    
    return fig

def create_solution_plots(solver):
    """Create solution plots with PINN and RK4 comparison"""
    if not solver:
        return None, None, None
    
    # Generate time points for evaluation
    t_eval = np.linspace(st.session_state.params.t_start, 
                        st.session_state.params.t_end, 
                        st.session_state.params.n_points)
    
    # Get PINN predictions
    predictions = solver.predict(t_eval)
    T_pred = predictions[:, 0]
    D_pred = predictions[:, 1]
    S_pred = predictions[:, 2]
    
    # Get RK4 solution for comparison
    t_rk4, solution_rk4 = runge_kutta_4th_order(st.session_state.params)
    T_rk4 = solution_rk4[:, 0]
    D_rk4 = solution_rk4[:, 1]
    S_rk4 = solution_rk4[:, 2]
    
    # Individual plots with comparison
    fig_T = go.Figure()
    fig_T.add_trace(go.Scatter(x=t_eval, y=T_pred, mode='lines', 
                              name='PINN Solution', line=dict(color='red', width=3)))
    fig_T.add_trace(go.Scatter(x=t_rk4, y=T_rk4, mode='lines', 
                              name='RK4 Solution', line=dict(color='darkred', width=2, dash='dash')))
    fig_T.update_layout(title="🦠 Tumor Size Evolution (T)", 
                       xaxis_title="Time (days)", 
                       yaxis_title="Tumor Size (cells)",
                       height=400, showlegend=True,
                       plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)')
    
    fig_D = go.Figure()
    fig_D.add_trace(go.Scatter(x=t_eval, y=D_pred, mode='lines', 
                              name='PINN Solution', line=dict(color='blue', width=3)))
    fig_D.add_trace(go.Scatter(x=t_rk4, y=D_rk4, mode='lines', 
                              name='RK4 Solution', line=dict(color='darkblue', width=2, dash='dash')))
    fig_D.update_layout(title="🧬 Dystrophin Level Evolution (D)", 
                       xaxis_title="Time (days)", 
                       yaxis_title="Dystrophin Level (units)",
                       height=400, showlegend=True,
                       plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)')
    
    fig_S = go.Figure()
    fig_S.add_trace(go.Scatter(x=t_eval, y=S_pred, mode='lines', 
                              name='PINN Solution', line=dict(color='green', width=3)))
    fig_S.add_trace(go.Scatter(x=t_rk4, y=S_rk4, mode='lines', 
                              name='RK4 Solution', line=dict(color='darkgreen', width=2, dash='dash')))
    fig_S.update_layout(title="📊 Age/Staging Impact Evolution (S)", 
                       xaxis_title="Time (days)", 
                       yaxis_title="Staging Impact (units)",
                       height=400, showlegend=True,
                       plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)')
    
    return fig_T, fig_D, fig_S

def create_phase_plots(solver):
    """Create phase space plots with PINN and RK4 comparison"""
    if not solver:
        return None
    
    t_eval = np.linspace(st.session_state.params.t_start, 
                        st.session_state.params.t_end, 
                        st.session_state.params.n_points)
    
    # PINN predictions
    predictions = solver.predict(t_eval)
    T_pred = predictions[:, 0]
    D_pred = predictions[:, 1]
    S_pred = predictions[:, 2]
    
    # RK4 solution
    t_rk4, solution_rk4 = runge_kutta_4th_order(st.session_state.params)
    T_rk4 = solution_rk4[:, 0]
    D_rk4 = solution_rk4[:, 1]
    S_rk4 = solution_rk4[:, 2]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('T vs D Phase Plot', 'D vs S Phase Plot', 
                       'S vs T Phase Plot', '3D Trajectory'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter3d"}]]
    )
    
    # T vs D
    fig.add_trace(
        go.Scatter(x=T_pred, y=D_pred, mode='lines', 
                  name='PINN T-D', line=dict(color='purple', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=T_rk4, y=D_rk4, mode='lines', 
                  name='RK4 T-D', line=dict(color='darkmagenta', width=2, dash='dash')),
        row=1, col=1
    )
    
    # D vs S
    fig.add_trace(
        go.Scatter(x=D_pred, y=S_pred, mode='lines', 
                  name='PINN D-S', line=dict(color='orange', width=3)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=D_rk4, y=S_rk4, mode='lines', 
                  name='RK4 D-S', line=dict(color='darkorange', width=2, dash='dash')),
        row=1, col=2
    )
    
    # S vs T
    fig.add_trace(
        go.Scatter(x=S_pred, y=T_pred, mode='lines', 
                  name='PINN S-T', line=dict(color='cyan', width=3)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=S_rk4, y=T_rk4, mode='lines', 
                  name='RK4 S-T', line=dict(color='darkcyan', width=2, dash='dash')),
        row=2, col=1
    )
    
    # 3D trajectory
    fig.add_trace(
        go.Scatter3d(x=T_pred, y=D_pred, z=S_pred, mode='lines',
                    name='PINN 3D', line=dict(color='magenta', width=5)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter3d(x=T_rk4, y=D_rk4, z=S_rk4, mode='lines',
                    name='RK4 3D', line=dict(color='darkviolet', width=3)),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True,
                     plot_bgcolor='rgba(0,0,0,0)',
                     paper_bgcolor='rgba(0,0,0,0)')
    
    return fig

def render_training_progress():
    """Render training progress section"""
    if st.session_state.training_active and st.session_state.solver:
        st.subheader("📈 Training Progress")
        
        # Progress bar
        progress = st.session_state.current_epoch / st.session_state.total_epochs
        st.progress(progress)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Epoch", f"{st.session_state.current_epoch}/{st.session_state.total_epochs}")
        
        solver = st.session_state.solver
        if solver.loss_history['total']:
            with col2:
                st.metric("Total Loss", f"{solver.loss_history['total'][-1]:.2e}")
            with col3:
                st.metric("Physics Loss", f"{solver.loss_history['physics'][-1]:.2e}")
            with col4:
                st.metric("IC Loss", f"{solver.loss_history['ic'][-1]:.2e}")
        
        # Live loss plots
        loss_fig = create_loss_plots(solver)
        if loss_fig:
            st.plotly_chart(loss_fig, use_container_width=True)

def training_loop(solver, epochs, ic_weight, progress_placeholder, metrics_placeholder):
    """Training loop that updates UI"""
    st.session_state.training_active = True
    st.session_state.current_epoch = 0
    
    for epoch in range(epochs):
        if not st.session_state.training_active:
            break
            
        # Training step
        total_loss, physics_loss, ic_loss = solver.train_step(ic_weight)
        
        # Store loss history
        solver.loss_history['total'].append(float(total_loss))
        solver.loss_history['physics'].append(float(physics_loss))
        solver.loss_history['ic'].append(float(ic_loss))

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Total Loss: {total_loss:.4e}, "
              f"Physics Loss: {physics_loss:.4e}, "
              f"IC Loss: {ic_loss:.4e}")
        
        st.session_state.current_epoch = epoch + 1
        
        # Update UI every 50 epochs
        if epoch % 50 == 0:
            time.sleep(0.01)  # Small delay to prevent UI freezing
    
    st.session_state.training_active = False

# Main application
def main():
    """Main application function"""
    initialize_session_state()
    
    # Title
    st.title("🧬 PINN-Based ODE Solver for Tumor-Dystrophin-Staging System")
    st.markdown("---")
    
    # Sidebar
    layers, activation, lr, epochs, ic_weight = render_sidebar()
    
    # Training controls
    start_training, stop_training, reset_params, save_current = render_training_controls()
    
    # Handle button clicks
    if start_training and not st.session_state.training_active:
        # Initialize solver
        st.session_state.solver = PINNSolver(
            st.session_state.params, layers, activation, lr
        )
        st.session_state.total_epochs = epochs
        
        # Start training
        with st.spinner("Initializing training..."):
            # Run training loop
            progress_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            training_loop(st.session_state.solver, epochs, ic_weight, 
                         progress_placeholder, metrics_placeholder)
        
        st.success("Training completed!")
        st.rerun()
    
    if stop_training:
        st.session_state.training_active = False
        st.warning("Training stopped by user.")
        st.rerun()
    
    if reset_params:
        st.session_state.params = ODEParameters()
        st.success("Parameters reset to defaults.")
        st.rerun()
    
    if save_current and st.session_state.solver:
        model_name = st.text_input("Model name", "pinn_model")
        if model_name:
            filename = save_model(st.session_state.solver, model_name, st.session_state.params)
            st.success(f"Model saved as {filename}")
    
    # Render training progress
    render_training_progress()
    
    # Results section
    if st.session_state.solver and not st.session_state.training_active:
        st.subheader("📊 Results")
        
        # Plot layout options
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.plot_layout = st.selectbox(
                "Plot Layout", ['vertical', 'horizontal', 'grid']
            )
        with col2:
            st.session_state.show_combined = st.checkbox("Show Combined View")
        with col3:
            show_phase = st.checkbox("Show Phase Plots")
        
        # Solution plots
        fig_T, fig_D, fig_S = create_solution_plots(st.session_state.solver)
        
        if st.session_state.plot_layout == 'vertical':
            st.plotly_chart(fig_T, use_container_width=True)
            st.plotly_chart(fig_D, use_container_width=True)
            st.plotly_chart(fig_S, use_container_width=True)
        elif st.session_state.plot_layout == 'horizontal':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(fig_T, use_container_width=True)
            with col2:
                st.plotly_chart(fig_D, use_container_width=True)
            with col3:
                st.plotly_chart(fig_S, use_container_width=True)
        else:  # grid
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_T, use_container_width=True)
                st.plotly_chart(fig_S, use_container_width=True)
            with col2:
                st.plotly_chart(fig_D, use_container_width=True)
        
        # Loss Function Plots
        st.subheader("📉 Loss Function Analysis")
        st.markdown("Training loss evolution showing how well the PINN learned the physics and initial conditions:")
        
        loss_fig = create_loss_plots(st.session_state.solver)
        if loss_fig:
            st.plotly_chart(loss_fig, use_container_width=True)
        else:
            st.info("No loss history available. Train the model to see loss plots.")
        
        # Phase plots
        if show_phase:
            st.subheader("🌀 Phase Space Analysis")
            phase_fig = create_phase_plots(st.session_state.solver)
            if phase_fig:
                st.plotly_chart(phase_fig, use_container_width=True)
        
        # Comparison metrics
        st.subheader("📈 PINN vs RK4 Comparison")
        
        # Calculate comparison metrics
        t_eval = np.linspace(st.session_state.params.t_start, 
                            st.session_state.params.t_end, 
                            st.session_state.params.n_points)
        
        # PINN solution
        pinn_predictions = st.session_state.solver.predict(t_eval)
        
        # RK4 solution
        t_rk4, rk4_solution = runge_kutta_4th_order(st.session_state.params)
        
        # Calculate errors (interpolate RK4 to match PINN time points if needed)
        if len(t_rk4) == len(t_eval):
            T_error = np.mean(np.abs(pinn_predictions[:, 0] - rk4_solution[:, 0]))
            D_error = np.mean(np.abs(pinn_predictions[:, 1] - rk4_solution[:, 1]))
            S_error = np.mean(np.abs(pinn_predictions[:, 2] - rk4_solution[:, 2]))
            
            # Relative errors
            T_rel_error = np.mean(np.abs(pinn_predictions[:, 0] - rk4_solution[:, 0]) / (np.abs(rk4_solution[:, 0]) + 1e-10)) * 100
            D_rel_error = np.mean(np.abs(pinn_predictions[:, 1] - rk4_solution[:, 1]) / (np.abs(rk4_solution[:, 1]) + 1e-10)) * 100
            S_rel_error = np.mean(np.abs(pinn_predictions[:, 2] - rk4_solution[:, 2]) / (np.abs(rk4_solution[:, 2]) + 1e-10)) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("T(t) Mean Abs Error", f"{T_error:.2e}")
                st.metric("T(t) Mean Rel Error", f"{T_rel_error:.2f}%")
            with col2:
                st.metric("D(t) Mean Abs Error", f"{D_error:.2e}")
                st.metric("D(t) Mean Rel Error", f"{D_rel_error:.2f}%")
            with col3:
                st.metric("S(t) Mean Abs Error", f"{S_error:.2e}")
                st.metric("S(t) Mean Rel Error", f"{S_rel_error:.2f}%")
    
    # Information sections
    with st.expander("📖 Mathematical Model Information"):
        st.latex(r'''
        \begin{align}
        \frac{dT}{dt} &= rT\left(1 - \frac{T}{K}\right) - fDT \\
        \frac{dD}{dt} &= -kD + gSD \\
        \frac{dS}{dt} &= h(S_0 - S) - jDS
        \end{align}
        ''')
        
        st.markdown("""
        **Parameter Descriptions:**
        - **r**: Tumor growth rate (1/day)
        - **K**: Carrying capacity (cells)
        - **f**: Dystrophin impact on tumor (1/day)
        - **k**: Dystrophin degradation rate (1/day)
        - **g**: Dystrophin activation by staging (1/day)
        - **h**: Recovery rate from staging effects (1/day)
        - **S₀**: Reference staging level (units)
        - **j**: Feedback strength between dystrophin and staging (1/day)
        """)
    
    with st.expander("💡 How to Use"):
        st.markdown("""
        1. **Adjust Parameters**: Use the sidebar to modify ODE parameters, initial conditions, and PINN settings
        2. **Start Training**: Click "Start Training" to begin the PINN optimization process
        3. **Monitor Progress**: Watch real-time loss plots and training metrics
        4. **View Results**: After training, explore the solution plots and phase space analysis
        5. **Compare Solutions**: The plots show both PINN (solid lines) and RK4 (dashed lines) solutions for validation
        6. **Save Models**: Save trained models for later use or comparison
        
        **Tips:**
        - Start with default parameters for stable training
        - Increase epochs for better accuracy
        - Adjust IC weight if initial conditions are not well satisfied
        - Use different activation functions for various behaviors
        - Compare PINN vs RK4 errors to assess PINN accuracy
        
        **About the Comparison:**
        - **RK4 (Runge-Kutta 4th Order)**: Classical numerical method providing reference solution
        - **PINN**: Physics-informed neural network learning to satisfy the differential equations
        - **Error Metrics**: Absolute and relative errors between PINN and RK4 solutions
        
        **Interface:**
        - **Clean Design**: Light interface with white backgrounds optimized for scientific computing
        - **Responsive Layout**: Adaptive design that works on different screen sizes
        - **Config File**: Modify `.streamlit/config.toml` for custom theme settings if needed
        """)
    
    with st.expander("🧠 About Physics-Informed Neural Networks"):
        st.markdown("""
        Physics-Informed Neural Networks (PINNs) combine the power of neural networks with physical laws
        encoded as differential equations. Key advantages:
        
        - **Physical Consistency**: Solutions automatically satisfy the governing equations
        - **Data Efficiency**: Can work with limited or no training data
        - **Flexibility**: Handle complex geometries and boundary conditions
        - **Uncertainty Quantification**: Provide measures of solution confidence
        
        In this application, PINNs solve the tumor-dystrophin-staging system by minimizing both
        the differential equation residuals and initial condition errors.
        """)

if __name__ == "__main__":
    main()