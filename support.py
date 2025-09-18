
import math
import numpy as np
from Classes.Body import Body

# ------------------ User parameters ------------------
WIDTH, HEIGHT = 1200, 800
BACKGROUND_COLOR = (8, 12, 25)
TRAIL_ALPHA = 180
G = 1.0  # gravitational constant in code units
INITIAL_TIME_SCALE = 1.0
PHYSICS_DT = 1e-3  # base physics timestep (simulation units)
MAX_SUBSTEPS = 10
TRAIL_MAX_LEN = 2000
SOFTENING = 1e-4  # prevents singularities in close encounters


# ------------------ Example system creators ------------------
def create_two_body_star_planet():

    m1 = 1.0
    m2 = 0.001
    r0 = 1.0
    v_circ = math.sqrt(G * (m1 + m2) / r0)

    b1 = Body(m1, np.array([0.0, 0.0], dtype=float), np.array([0.0, 0.0], dtype=float),
              radius=8, color=(255, 200, 50))
    b2 = Body(m2, np.array([r0, 0.0], dtype=float), np.array([0.0, 0.95 * v_circ], dtype=float),
              radius=5, color=(120, 170, 255))
    
    return [b1, b2]

def create_five_body_demo():
    # small compact system of several bodies (toy)
    bodies = []
    np.random.seed(42)
    center_mass = 2.0
    bodies.append(Body(center_mass, np.array([0.0, 0.0]), np.array([0.0, 0.0]), radius=10, color=(255,180,80)))

    for i in range(4):
        angle = i * (2.0 * math.pi / 4.0)
        r = 0.8 + 0.2 * i
        pos = np.array([r * math.cos(angle), r * math.sin(angle)])
        speed = math.sqrt(G * center_mass / r) * (0.9 + 0.05 * i)
        vel = np.array([-math.sin(angle) * speed, math.cos(angle) * speed])
        bodies.append(Body(0.02 + 0.01 * i, pos, vel, radius=4 + i, color=(80 + 40*i, 160, 220 - 30*i)))
        
    return bodies