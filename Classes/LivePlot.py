import numpy as np
from Classes.NBodySimulation import NBodySimulation
from Classes.Camera import Camera

# ------------------ Matplotlib live plot helper (optional) ------------------
class LivePlot:
    def __init__(self):
        self.enabled = False
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.fig = None

        except Exception:
            self.plt = None
            self.fig = None

    def toggle(self):
        if self.plt is None:
            print("matplotlib not available; install it to use live plots.")
            return
        
        if not self.enabled:
            # create figure
            self.fig, (self.ax_traj, self.ax_energy) = self.plt.subplots(1, 2, figsize=(12, 5))
            self.enabled = True
            self.plt.ion()
            self.plt.show()

        else:
            try:
                self.plt.close(self.fig)

            except Exception:
                pass
            self.fig = None
            self.enabled = False

    def update(self, sim: NBodySimulation, cam: Camera):
        if not self.enabled or self.plt is None:
            return
        
        # Trajectory plot (world coordinates, show trails for each body)
        self.ax_traj.cla()

        for b in sim.bodies:
            if len(b.trail) > 1:
                arr = np.array(b.trail)
                self.ax_traj.plot(arr[:, 0], arr[:, 1], '-', linewidth=1, alpha=0.9, color=np.array(b.color)/255.0)

            # draw current position
            self.ax_traj.scatter([b.pos[0]], [b.pos[1]], s=max(10, b.radius/1e-0), color=np.array(b.color)/255.0)

        self.ax_traj.set_title("Trajectories (world coords)")
        self.ax_traj.set_xlabel("x")
        self.ax_traj.set_ylabel("y")
        self.ax_traj.set_aspect('equal', adjustable='box')
        self.ax_traj.grid(True, alpha=0.2)

        # Energy plot
        self.ax_energy.cla()
        times = sim.time_history
        energies = sim.energy_history

        if len(times) > 0 and len(energies) > 0:
            E0 = energies[0]
            rel = [(E - E0) / abs(E0) if abs(E0) > 0 else (E - E0) for E in energies]
            self.ax_energy.plot(times, rel, '-')
            self.ax_energy.set_xlabel("time")
            self.ax_energy.set_ylabel("relative energy change")
            self.ax_energy.grid(True, alpha=0.2)
            self.ax_energy.set_title("Energy conservation (relative change)")
            
        else:
            self.ax_energy.text(0.5, 0.5, "No energy data yet", ha='center')

        self.plt.pause(0.001)