import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Quantum Mechanical Carnot Engine (Bender et al.)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS for Formal Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #000000;
    }
    .stMarkdown {
        font-family: 'Georgia', serif;
        color: #333333;
    }
    .metric-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Title and Introduction ---
st.title("Quantum Mechanical Carnot Engine")
st.markdown("""
**Reference:** C.M. Bender, D.C. Brody, and B.K. Meister, *Quantum mechanical Carnot engine*, arXiv:quant-ph/0007002 (2000).

This simulation implements the exact cycle described in the paper. Unlike a classical engine, the efficiency of this single-particle quantum engine depends solely on the geometry of the potential well (the ratio of expansion), not on temperature.

**The Cycle Stages:**
1.  **Isoenergetic Expansion ($1 \\to 2$):** Constant expectation energy $\\langle E \\rangle = E_H$. System transitions from ground state $u_1$ to excited state $u_2$.
2.  **Adiabatic Expansion ($2 \\to 3$):** System stays in pure state $u_2$.
3.  **Isoenergetic Compression ($3 \\to 4$):** Constant expectation energy $\\langle E \\rangle = E_L$. System transitions from $u_2$ back to $u_1$.
4.  **Adiabatic Compression ($4 \\to 1$):** System stays in pure state $u_1$.
""")

st.divider()

# --- Simulation Logic ---

def quantum_carnot_cycle(L_A, L_C):
    """
    Calculates the path for the Bender-Brody-Meister Quantum Carnot Cycle.
    
    Parameters:
    L_A (float): Starting width (L_min). Corresponds to state u1.
    L_C (float): Maximum width (L_max). Corresponds to state u2.
    
    Constraints from paper:
    L_B = 2 * L_A
    L_D = L_C / 2
    Validity: L_C must be > 2 * L_A (to ensure L_C > L_B).
    """
    
    # Constants (Set to 1 for normalized visualization as per paper logic)
    # The paper often uses units where h_bar^2 * pi^2 / m = 1 for simplicity in derived formulas
    # F scales with this constant, but the shape of the graph is invariant.
    K = 1.0 
    
    # --- Check Validity ---
    if L_C <= 2 * L_A:
        return None, "Violation of quantum condition: L_max must be > 2 * L_min."

    # --- Points ---
    L_B = 2 * L_A
    L_D = L_C / 2.0
    
    points = {'A': L_A, 'B': L_B, 'C': L_C, 'D': L_D}
    
    # --- Path 1: A -> B (Isoenergetic Expansion at E_H) ---
    # From paper: f_AB(L) = (pi^2 h^2) / (m * L_A^2 * L)
    # Scaled: F = K / (L_A^2 * L)
    # Energy is constant E_H = K / (2 * L_A^2)
    l1 = np.linspace(L_A, L_B, 50)
    f1 = (K / (L_A**2)) * (1.0 / l1)
    
    # --- Path 2: B -> C (Adiabatic Expansion in state u2) ---
    # From paper: f_BC(L) = (4 * pi^2 h^2) / (m * L^3)
    # Scaled: F = 4K / L^3
    l2 = np.linspace(L_B, L_C, 50)
    f2 = 4 * K / (l2**3)
    
    # --- Path 3: C -> D (Isoenergetic Compression at E_L) ---
    # From paper: f_CD(L) = (4 * pi^2 h^2) / (m * L_C^2 * L)
    # Scaled: F = 4K / (L_C^2 * L)
    # Energy is constant E_L = 4K / (2 * L_C^2)
    l3 = np.linspace(L_C, L_D, 50)
    f3 = (4 * K / (L_C**2)) * (1.0 / l3)
    
    # --- Path 4: D -> A (Adiabatic Compression in state u1) ---
    # From paper: f_DA(L) = (pi^2 h^2) / (m * L^3)
    # Scaled: F = K / L^3
    l4 = np.linspace(L_D, L_A, 50)
    f4 = K / (l4**3)
    
    # Combine paths
    L_cycle = np.concatenate([l1, l2, l3, l4])
    F_cycle = np.concatenate([f1, f2, f3, f4])
    
    # --- Work Calculation ---
    # Analytical integrals from paper to ensure precision
    # W_AB = (K / L_A^2) * ln(L_B/L_A) = (K/L_A^2) * ln(2)
    W_AB = (K / L_A**2) * np.log(2)
    
    # W_BC = Int(4K/L^3) from L_B to L_C = [-2K/L^2] from L_B to L_C
    #      = 2K * (1/L_B^2 - 1/L_C^2)
    W_BC = 2 * K * ((1.0/L_B**2) - (1.0/L_C**2))
    
    # W_CD = (4K / L_C^2) * ln(L_D/L_C) = (4K/L_C^2) * ln(0.5) = -(4K/L_C^2)*ln(2)
    W_CD = (4 * K / L_C**2) * np.log(0.5)
    
    # W_DA = Int(K/L^3) from L_D to L_A = [-K/(2L^2)] from L_D to L_A
    #      = (K/2) * (1/L_D^2 - 1/L_A^2)
    W_DA = (K / 2.0) * ((1.0/L_D**2) - (1.0/L_A**2))
    
    W_net = W_AB + W_BC + W_CD + W_DA
    
    # Efficiency Formula from Eq (7) in paper: eta = 1 - 4 * (L_A / L_C)^2
    efficiency = 1 - 4 * (L_A / L_C)**2
    
    return {
        'L': L_cycle,
        'F': F_cycle,
        'points': points,
        'W_net': W_net,
        'Efficiency': efficiency,
        'W_steps': (W_AB, W_BC, W_CD, W_DA)
    }, None


# --- User Controls ---
st.sidebar.header("Cycle Parameters")

# Inputs
L_min = st.sidebar.slider("Minimum Width (L_min = L_A)", 1.0, 5.0, 2.0, 0.1)
L_max_limit = 4.0 * L_min # Ensure slider range allows valid cycles
L_max = st.sidebar.slider("Maximum Width (L_max = L_C)", 2.0*L_min + 0.1, 10.0*L_min, 3.0*L_min, 0.1)

# Speed
speed = st.sidebar.slider("Simulation Speed", 0.1, 3.0, 1.0)
sleep_time = 0.05 / speed

# --- Calculation ---
data, error = quantum_carnot_cycle(L_min, L_max)

if error:
    st.error(error)
    st.stop()

# --- Main Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    # --- Plotting ---
    plot_placeholder = st.empty()

with col2:
    st.markdown("### Theoretical Metrics")
    
    eff = data['Efficiency']
    work = data['W_net']
    
    st.metric("Quantum Efficiency (η)", f"{eff:.4f}")
    st.metric("Net Work Output (Arbitrary Units)", f"{work:.4f}")
    
    st.markdown(r"""
    **Efficiency Formula:**
    $$ \eta = 1 - 4 \left( \frac{L_{min}}{L_{max}} \right)^2 $$
    
    **Work Stages:**
    * $1 \to 2$: Isoenergetic Expansion
    * $2 \to 3$: Adiabatic Expansion
    * $3 \to 4$: Isoenergetic Compression
    * $4 \to 1$: Adiabatic Compression
    """)

# --- Animation Loop ---
L_vals = data['L']
F_vals = data['F']
pts = data['points']

# Create a static background plot first to save resources
fig_bg, ax = plt.subplots(figsize=(8, 6))
ax.plot(L_vals, F_vals, color='#2c3e50', linewidth=1.5, alpha=0.3, label='Cycle Path')
ax.scatter([pts['A'], pts['B'], pts['C'], pts['D']], 
           [F_vals[0], F_vals[49], F_vals[99], F_vals[149]], 
           color='black', zorder=3)
ax.text(pts['A'], F_vals[0], '  1 (Start)', va='bottom')
ax.text(pts['B'], F_vals[49], '  2', va='bottom')
ax.text(pts['C'], F_vals[99], '  3', va='top')
ax.text(pts['D'], F_vals[149], '  4', va='top')

ax.set_xlabel("Well Width L (Arbitrary Units)")
ax.set_ylabel("Force F (Arbitrary Units)")
ax.set_title("Force vs Width Diagram")
ax.grid(True, linestyle='--', alpha=0.5)

# Interactive run
start_sim = st.button("Run Simulation", type="primary")

if start_sim:
    progress_bar = st.progress(0)
    
    for i in range(0, len(L_vals), 2): # Step by 2 for speed
        # Copy the background figure logic
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Redraw background
        ax.plot(L_vals, F_vals, color='#bdc3c7', linewidth=1, linestyle='--')
        ax.scatter([pts['A'], pts['B'], pts['C'], pts['D']], 
                   [F_vals[0], F_vals[49], F_vals[99], F_vals[149]], 
                   color='black', s=20)
        
        # Draw active segment
        ax.plot(L_vals[:i+1], F_vals[:i+1], color='#e74c3c', linewidth=2.5)
        
        # Current Point
        ax.scatter(L_vals[i], F_vals[i], color='#c0392b', s=100, zorder=5)
        
        # Labels
        ax.set_xlabel("Well Width L")
        ax.set_ylabel("Force F")
        ax.set_title("Quantum Carnot Cycle: Force vs Width")
        ax.grid(True, alpha=0.3)
        
        # Text annotation for phase
        if i < 50:
            phase = "Isoenergetic Expansion"
        elif i < 100:
            phase = "Adiabatic Expansion"
        elif i < 150:
            phase = "Isoenergetic Compression"
        else:
            phase = "Adiabatic Compression"
            
        ax.text(0.95, 0.95, phase, transform=ax.transAxes, ha='right', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        
        progress_bar.progress(i / len(L_vals))
        time.sleep(sleep_time)

    st.success("Simulation Complete.")
else:
    plot_placeholder.pyplot(fig_bg)
