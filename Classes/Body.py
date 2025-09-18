import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ------------------ Data classes ------------------
@dataclass
class Body:
    mass: float
    pos: np.ndarray         # shape (2,)
    vel: np.ndarray         # shape (2,)
    radius: float = 5.0
    color: Tuple[int, int, int] = (255, 255, 255)
    trail: List[np.ndarray] = field(default_factory=list)

    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * float(np.dot(self.vel, self.vel))