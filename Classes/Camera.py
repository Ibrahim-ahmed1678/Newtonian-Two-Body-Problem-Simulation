import numpy as np
from typing import Tuple

# ------------------ Camera / Utilities ------------------
class Camera:
    def __init__(self, width, height, scale=200.0):
        self.width = width
        self.height = height
        self.scale = float(scale)  # pixels per simulation unit
        self.offset = np.array([0.0, 0.0], dtype=float)
        self.dragging = False
        self.last_mouse = None

    def world_to_screen(self, pos: np.ndarray) -> Tuple[int, int]:
        cx, cy = self.width // 2, self.height // 2
        x = cx + (pos[0] - self.offset[0]) * self.scale
        y = cy - (pos[1] - self.offset[1]) * self.scale

        return int(round(x)), int(round(y))

    def screen_to_world(self, sx: int, sy: int) -> np.ndarray:
        cx, cy = self.width // 2, self.height // 2
        x = (sx - cx) / self.scale + self.offset[0]
        y = -(sy - cy) / self.scale + self.offset[1]

        return np.array([x, y], dtype=float)

    def start_drag(self, mx, my):
        self.dragging = True
        self.last_mouse = (mx, my)

    def drag(self, mx, my):
        if not self.dragging or self.last_mouse is None:
            return
        
        dx = (mx - self.last_mouse[0]) / self.scale
        dy = (my - self.last_mouse[1]) / self.scale
        
        # dragging moves the world offset inversely
        self.offset -= np.array([dx, -dy])
        self.last_mouse = (mx, my)

    def end_drag(self):
        self.dragging = False
        self.last_mouse = None