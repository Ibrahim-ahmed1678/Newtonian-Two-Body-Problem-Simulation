import math
import numpy as np
from typing import List
from Classes.Body import Body
from support import G, SOFTENING



# ------------------ Simulation Core (N-body) ------------------
class NBodySimulation:
    def __init__(self, bodies: List[Body], G_const: float = G, soft: float = SOFTENING):
        self.bodies = bodies
        self.G = G_const
        self.softening = soft
        self.time = 0.0

        # Compute initial accelerations for velocity verlet
        self.accs = self.compute_accelerations()
        self.initial_state = self._snapshot_state()
        self.energy_history: List[float] = []
        self.time_history: List[float] = []

    def _snapshot_state(self):
        return {
            "time": self.time,
            "bodies": [(b.pos.copy(), b.vel.copy(), list(b.trail)) for b in self.bodies]
        }

    def reset(self):
        s = self.initial_state
        self.time = float(s["time"])

        for b, (pos, vel, trail) in zip(self.bodies, s["bodies"]):
            b.pos = pos.copy()
            b.vel = vel.copy()
            b.trail = [p.copy() for p in trail]

        self.accs = self.compute_accelerations()
        self.energy_history.clear()
        self.time_history.clear()

    def compute_accelerations(self) -> np.ndarray:
        """Compute N accelerations (shape: (N,2)) due to mutual gravity with softening."""
        N = len(self.bodies)
        accs = np.zeros((N, 2), dtype=float)
        positions = np.array([b.pos for b in self.bodies], dtype=float)
        masses = np.array([b.mass for b in self.bodies], dtype=float)

        # pairwise loops (N small in these demos — vectorization possible but loops are clear)
        for i in range(N):
            pi = positions[i]
            ai = np.zeros(2, dtype=float)

            for j in range(N):
                if i == j:
                    continue

                r = positions[j] - pi
                dist2 = float(np.dot(r, r)) + self.softening
                inv_dist3 = 1.0 / (math.sqrt(dist2) * dist2)
                ai += self.G * masses[j] * r * inv_dist3

            accs[i] = ai

        return accs

    def velocity_verlet_step(self, dt: float):
        """Advance the system by dt using velocity Verlet for N bodies."""
        N = len(self.bodies)

        # current accelerations in self.accs
        # update positions
        for i, b in enumerate(self.bodies):
            b.pos = b.pos + b.vel * dt + 0.5 * self.accs[i] * (dt * dt)

        # compute new accelerations
        new_accs = self.compute_accelerations()

        # update velocities
        for i, b in enumerate(self.bodies):
            b.vel = b.vel + 0.5 * (self.accs[i] + new_accs[i]) * dt

        # commit
        self.accs = new_accs
        self.time += dt

    def total_energy(self) -> float:
        """Compute total energy = KE + PE (pairwise)."""
        KE = sum(b.kinetic_energy() for b in self.bodies)
        PE = 0.0
        N = len(self.bodies)

        # pairwise potential
        for i in range(N):
            for j in range(i + 1, N):
                r = self.bodies[j].pos - self.bodies[i].pos
                dist = math.sqrt(float(np.dot(r, r)) + self.softening)
                PE += -self.G * self.bodies[i].mass * self.bodies[j].mass / dist
                
        return KE + PE
