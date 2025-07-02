# 🧬 PINN-Based ODE Solver for Tumor-Dystrophin-Staging System

A Physics-Informed Neural Network (PINN) implementation for solving complex systems of ordinary differential equations (ODEs), specifically designed for modeling tumor-dystrophin-staging dynamics with comparative analysis against traditional numerical methods.

## 🚀 Quick Start

### Prerequisites

Make sure you have Python 3.8+ installed, then install the required dependencies:

```bash
pip install streamlit tensorflow numpy matplotlib plotly pandas
```

### Running the Application

1. **Clone or navigate to the project directory**
2. **Launch the Streamlit app:**
   ```bash
   streamlit run final.py
   ```
3. **Open your browser** - Streamlit will automatically open `http://localhost:8501`

The application will start with a clean interface optimized for scientific computing visualization.

## 📊 What This Application Does

This PINN solver addresses a sophisticated **tumor-dystrophin-staging system** modeled by the following coupled ODEs:

```
dT/dt = rT(1 - T/K) - fDT     (Tumor dynamics)
dD/dt = -kD + gSD             (Dystrophin regulation)  
dS/dt = h(S₀ - S) - jDS       (Staging progression)
```

### Key Features

- **🧠 Physics-Informed Neural Networks**: Leverages deep learning that respects physical laws
- **📈 Real-time Training Visualization**: Monitor loss functions and convergence in real-time
- **🔄 Method Comparison**: Direct comparison between PINN and Runge-Kutta 4th order solutions
- **🎛️ Interactive Parameter Control**: Adjust all system parameters dynamically
- **📊 Comprehensive Visualization**: Solution plots, phase space analysis, and error metrics
- **💾 Model Management**: Save and load trained models for reproducibility
- **📱 Responsive UI**: Clean, adaptive design that works on different screen sizes

### Application Interface

- **Parameter Panel**: Adjust ODE parameters (growth rates, degradation constants, etc.)
- **Network Configuration**: Customize PINN architecture (layers, activation functions, learning rates)
- **Training Controls**: Start/stop training, monitor progress, save models
- **Results Dashboard**: Interactive plots comparing PINN vs traditional methods
- **Phase Space Analysis**: Visualize system behavior in multi-dimensional space

## 🎯 Why This is Important for ODE Systems

### Traditional Challenges in ODE Solving

1. **Stiffness**: Many biological systems exhibit stiff equations that challenge standard solvers
2. **Parameter Sensitivity**: Small parameter changes can dramatically affect solutions  
3. **Computational Efficiency**: Classical methods may require very small time steps
4. **Boundary/Initial Conditions**: Complex systems often have intricate constraint requirements

### PINN Advantages

- **🔬 Physics Consistency**: Solutions automatically satisfy the governing differential equations
- **📊 Data Efficiency**: Can work with sparse or noisy experimental data
- **🎯 Constraint Handling**: Naturally incorporates physical constraints and conservation laws
- **🔄 Flexibility**: Adapts to complex geometries and irregular domains
- **⚡ Parallel Computing**: Leverages GPU acceleration for faster computation
- **🎲 Uncertainty Quantification**: Provides confidence measures for predictions

### Specific Benefits for This System

The tumor-dystrophin-staging system exhibits:
- **Nonlinear coupling** between biological processes
- **Multiple time scales** (fast dystrophin dynamics vs. slow tumor growth)
- **Regulatory feedback loops** that traditional methods may struggle to capture accurately

## 🔬 Comparison with Other Methods

This application enables direct comparison between:

### 1. **PINN (Physics-Informed Neural Networks)**
- **Strengths**: Physics consistency, handles complex boundary conditions, GPU acceleration
- **Best for**: Systems with known governing equations but complex parameter dependencies

### 2. **Runge-Kutta 4th Order (RK4)**
- **Strengths**: Well-established, reliable, explicit time-stepping
- **Best for**: Standard ODE systems with moderate stiffness

### 3. **Error Analysis Features**
- **Absolute Error**: Direct difference between PINN and RK4 solutions
- **Relative Error**: Percentage-based comparison accounting for solution magnitude
- **Visual Comparison**: Side-by-side plots highlighting differences
- **Convergence Metrics**: Track how PINN performance improves with training

### Future Comparison Capabilities

The framework is designed to easily incorporate additional methods:
- **Implicit solvers** (Backward Euler, BDF methods)
- **Adaptive step-size methods** (Dormand-Prince, Cash-Karp)
- **Specialized biological solvers** (Gillespie for stochastic systems)
- **Hybrid approaches** (PINN-assisted classical methods)

## 🛠️ Advanced Usage

### Parameter Tuning

1. **Start with defaults** for stable initial behavior
2. **Adjust network architecture** (more layers for complex dynamics)
3. **Tune learning rates** (lower for stability, higher for speed)
4. **Balance physics vs. IC losses** using the IC weight parameter

### Model Interpretation

- Monitor **physics loss** to ensure equation satisfaction
- Check **initial condition loss** for proper constraint handling
- Analyze **phase plots** for system stability and attractors
- Compare **error metrics** to assess PINN accuracy vs. classical methods

### Performance Optimization

- Use **GPU acceleration** with TensorFlow for larger networks
- **Batch training** for multiple parameter sets
- **Transfer learning** from similar ODE systems
- **Adaptive mesh refinement** for critical time regions

## 📈 Output Interpretation

The application provides multiple views of your ODE system:

- **Time Series**: Evolution of T(t), D(t), S(t) over time
- **Phase Space**: Trajectory visualization in 3D state space  
- **Error Analysis**: Quantitative comparison with reference solutions
- **Training Metrics**: Loss function evolution and convergence assessment

## 🔮 Research Applications

This PINN framework is applicable to various scientific domains:

- **Biomedical Modeling**: Tumor growth, drug kinetics, population dynamics
- **Engineering Systems**: Control theory, fluid dynamics, heat transfer
- **Climate Modeling**: Atmospheric dynamics, carbon cycle modeling
- **Economics**: Market dynamics, resource allocation models

---

*Built with Streamlit, TensorFlow, and modern scientific computing practices for robust, reproducible ODE solving with physics-informed machine learning.*
