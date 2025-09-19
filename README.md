# Two-Body Planetary Simulation

A Python-based physics engine and visualization toolkit for simulating Newtonian two-body gravitational dynamics in 2D.  
Features **adaptive timestepping**, **energy conservation checks**, **real-time Pygame visualization**, and **Matplotlib-based orbit analysis**.  

---

## Features
- **Physics Engine** — Simulates gravitational interactions with numerical integration via `scipy.solve_ivp`.  
-  **Adaptive Timestepping** — Maintains stability across fast and slow orbital regimes.  
-  **Energy Conservation Checks** — Tracks kinetic, potential, and total system energy.
  
- **Visualizations**  
  - **Pygame (real-time)** — Interactive orbit visualizer with trails, scaling, and live info.  
  - **Matplotlib (analysis)** — Detailed orbital trajectory plots and energy trends.
      
- **Predefined Systems** — Earth–Moon and Binary Star examples included.  
---

##  Installation
Install dependencies:
```bash
pip install -r requirements.txt
