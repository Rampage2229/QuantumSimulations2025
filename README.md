# QuantumSimulations2025
Simulations done for the quantum physics course presentation on UPC 2025-26 fall semester.


## Requirements
Python 3.14.0 and libraries on requirements.txt

You can easily install them by running:

```pip install requirements.txt```

## Simulations

### Carnot simulation

Execute ```streamlit run carnot_sim.py``` on your terminal after installing the required libraries. In your browser the simulation should automatically open on a new tab.

This program simulates the carnot cycle of a single-particle system on an infinite potential well of varying length. You may adjust the maximum and minimum size of the well, and get the theoretical efficiency of the machine, and a real-time view of the particle's eigenfunction.

### Thermalization simulation

Execute ```streamlit run thermalization_sim.py``` on your terminal after installing the required libraries. In your browser the simulation should automatically open on a new tab.

This program simulates an n-particle system with an starting spin configuration. After hitting play, or moving the time slider, a simuation plays on how the expected spin value of each particle changes according to quantum thermodynamics. There are graphs of information, and a real-time simulation of the reduced density matrix of the two middle spins. You can configure parameters such as interaction force, disroder, initial state and number of particles.
