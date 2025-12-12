import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qutip import *

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Quantum Lab: Numeric", layout="wide", page_icon="🔢")

st.markdown("""
<style>
    .main { background-color: #f5f7f9; }
    h1 { color: #2c3e50; }
    div[data-testid="stSidebarUserContent"] {
        background-color: #ffffff;
        border-right: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. PHYSICS ENGINE ---

@st.cache_data
def simulate_and_package_data(N, disorder_strength, interaction_strength, initial_state_choice):
    """
    Simulates the system and prepares TEXT grids for the matrix.
    """
    steps = 50 
    times = np.linspace(0, 15, steps)
    
    # Operators
    sx_list, sy_list, sz_list = [], [], []
    for i in range(N):
        op_list = [qeye(2)] * N
        op_list[i] = sigmax()
        sx_list.append(tensor(op_list))
        op_list[i] = sigmay()
        sy_list.append(tensor(op_list))
        op_list[i] = sigmaz()
        sz_list.append(tensor(op_list))

    # Hamiltonian
    H_int = 0
    for i in range(N - 1):
        H_int += -interaction_strength * (sx_list[i]*sx_list[i+1] + 
                                          sy_list[i]*sy_list[i+1] + 
                                          sz_list[i]*sz_list[i+1])
    
    np.random.seed(42) 
    random_fields = np.random.uniform(-disorder_strength, disorder_strength, N)
    H_dis = sum([random_fields[i] * sz_list[i] for i in range(N)])
    H = H_int + H_dis
    
    # Initial State: Néel State |↑↓↑↓...>
    psi_list = []
    
    if initial_state_choice == "Néel State (↑↓↑↓)":
        psi_list = [basis(2, 0) if i % 2 == 0 else basis(2, 1) for i in range(N)]
        
    elif initial_state_choice == "Domain Wall (↑↑↓↓)":
        # First half UP, Second half DOWN
        psi_list = [basis(2, 0) if i < N//2 else basis(2, 1) for i in range(N)]
        
    elif initial_state_choice == "Single Excitation (↑↓↓↓)":
        # First spin UP, rest DOWN
        psi_list = [basis(2, 0) if i == 0 else basis(2, 1) for i in range(N)]
        
    elif initial_state_choice == "Random Product":
        # Random binary choice for each site
        import random
        # We use a local random seed so it changes only when the user wants
        states = [basis(2, 0), basis(2, 1)]
        psi_list = [random.choice(states) for _ in range(N)]
    psi0 = tensor(psi_list)
    
    # Evolution
    result = sesolve(H, psi0, times)
    
    # Data Containers
    entropy_vals = []
    spin_vals = []
    density_matrices = [] # Numerical values for color
    density_text = []     # String values for display
    purities = []
    
    # We focus on the middle 2 spins (Reduced Density Matrix)
    target_spins = [N//2 - 1, N//2] 
    
    for state in result.states:
        # 1. Entropy
        rho_half = state.ptrace(list(range(N // 2)))
        entropy_vals.append(entropy_vn(rho_half, base=np.e))
        
        # 2. Spins
        spin_vals.append([expect(sz_list[i], state) for i in range(N)])
        
        # 3. Density Matrix
        rho_reduced = state.ptrace(target_spins)
        dm_data = np.abs(rho_reduced.full())
        
        # FIX 1: Do not flip data computationally. We will handle orientation in Plotly axes.
        density_matrices.append(dm_data)
        
        # Create a text grid for the matrix
        # formatting numbers to 2 decimal places
        text_grid = [[f"{val:.2f}" for val in row] for row in dm_data]
        density_text.append(text_grid)
        
        # 4. Purity
        purities.append((rho_reduced * rho_reduced).tr().real)

    return times, entropy_vals, spin_vals, density_matrices, density_text, purities

# --- 2. PLOTLY ANIMATION BUILDER ---

def build_animated_figure(times, entropy, spins, density_matrices, density_text, purities, N):
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Entanglement Entropy", 
            "Subsystem Purity", 
            "Spin Chain State", 
            "Reduced  Density Matrix (Middle 2 Spins)"
        ),
        vertical_spacing=0.2,
        horizontal_spacing=0.1
    )

    # --- TRACE 0: Entropy ---
    fig.add_trace(go.Scatter(x=[times[0]], y=[entropy[0]], mode='lines', 
                             line=dict(color='#e67e22', width=3), name="Entropy"), row=1, col=1)
    
    # --- TRACE 1: Purity ---
    fig.add_trace(go.Scatter(x=[times[0]], y=[purities[0]], mode='lines', 
                             line=dict(color='#2980b9', width=3), name="Purity"), row=1, col=2)

    # --- TRACE 2: Spin Heatmap ---
    fig.add_trace(go.Heatmap(
        z=[spins[0]], 
        x=[f"S{i+1}" for i in range(N)], y=["Spin Z"],
        colorscale="RdBu", zmin=-1, zmax=1, showscale=False
    ), row=2, col=1)

    # --- TRACE 3: Numeric Density Matrix ---
    # FIX 1 (Cont.): Standard labels top-to-bottom
    labels = ["↑↑", "↑↓", "↓↑", "↓↓"]
    
    fig.add_trace(go.Heatmap(
        z=density_matrices[0],
        text=density_text[0],  # THE TEXT LAYER
        texttemplate="%{text}", # Show the text
        # FIX 2: Changed text color to dark gray for better contrast against bright colors
        textfont={"size": 14, "color": "#FFFFFF"}, 
        x=labels, y=labels,
        colorscale="tealgrn", zmin=0, zmax=0.5, showscale=True # Hide colorbar to focus on numbers
    ), row=2, col=2)

    # --- FRAMES ---
    frames = []
    for k in range(len(times)):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=times[:k+1], y=entropy[:k+1]),
                go.Scatter(x=times[:k+1], y=purities[:k+1]),
                go.Heatmap(z=[spins[k]]),
                # Update BOTH z (color) and text (numbers)
                go.Heatmap(z=density_matrices[k], text=density_text[k])
            ],
            traces=[0, 1, 2, 3],
            name=str(k)
        ))

    fig.frames = frames

    # --- LAYOUT ---
    fig.update_layout(
        height=700,
        template="plotly_white",
        margin=dict(l=20, r=20, t=80, b=20),
        
        # Play Buttons
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            direction="right",
            x=-0.05, y=1.15,
            xanchor="left", yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
            ]
        )],
        sliders=[{
            "currentvalue": {"prefix": "Time: "},
            "pad": {"t": 50},
            "steps": [{"args": [[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], 
                       "label": f"{t:.1f}", "method": "animate"} for k, t in enumerate(times)]
        }]
    )
    
    
    # FIX 1 (Cont.): Fix Matrix Orientation (Top-Left = 0,0 index)
    fig.update_yaxes(autorange="reversed", row=2, col=2) 

    return fig

# --- 3. MAIN APP ---

st.sidebar.title(" Experiment")

state_options = [
    "Néel State (↑↓↑↓)", 
    "Domain Wall (↑↑↓↓)", 
    "Single Excitation (↑↓↓↓)", 
    "Random Product"
]
init_state = st.sidebar.selectbox("Initial State", state_options)

N_spins = st.sidebar.number_input("Number of particles", value = 6)
disorder = st.sidebar.slider("Disorder (Chaos)", 0.0, 5.0, 0.5)
interaction = st.sidebar.slider("Interaction Strength", 0.0, 2.0, 1.0)

st.title(" Quantum Lab: The Matrix")

# Run Sim
times, entropy, spins, dms, dm_text, purities = simulate_and_package_data(
    N_spins, disorder, interaction, init_state
)

# Build & Show
fig = build_animated_figure(times, entropy, spins, dms, dm_text, purities, N_spins)
st.plotly_chart(fig, use_container_width=True)

# Explanation Placeholder
st.markdown("---")
explanation_placeholder = st.empty()
# You can later use: explanation_placeholder.markdown("Your text here")


explanation_placeholder.markdown("Alex moral maricon")
