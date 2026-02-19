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
      
- **Predefined Systems** — Earth–Sun and 5 planet-Sun system examples included.  
---

##  Installation
Install dependencies:
```bash
pip install -r requirements.txt
```
---
## TODO:
- 5-body system diverges very quickly -> probably better to simply remove this part and instead simulate a two body system for each of the planets in our solar system.

- live trajectory plot is not that helpful -> perhaps a better feature would be to have phase-space plots (maybe).

- Simulating Mercury's orbit would be inaccurate (precession would not be accounted for) -> futher development would be to implement a general relativistic version that resolve this issue.

- current orbits are circular, which is ok for an approximation-> elliptical orbits would be more accurate.