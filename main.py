import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import pygame
import math
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class Body:
    """Represents a celestial body with position, velocity, and mass."""
    
    mass: float
    position: np.ndarray  # [x, y]
    velocity: np.ndarray  # [vx, vy]
    radius: float = 1.0
    color: Tuple[int, int, int] = (255, 255, 255)
    trail: List[np.ndarray] = None
    
    def __post_init__(self):
        if self.trail is None:
            self.trail = []
        self.position = np.array(self.position, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)

class PhysicsEngine:
    """Core physics engine for two-body gravitational simulation."""
    
    def __init__(self, G: float = 6.67430e-11):
        self.G = G  # Gravitational constant
        self.total_energy_history = []
        self.time_history = []
        
    def gravitational_force(self, body1: Body, body2: Body) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate gravitational force between two bodies."""

        r_vec = body2.position - body1.position
        r_mag = np.linalg.norm(r_vec)
        
        if r_mag == 0:
            return np.zeros(2), np.zeros(2)
        
        r_unit = r_vec / r_mag
        force_mag = self.G * body1.mass * body2.mass / (r_mag ** 2)
        
        force_on_1 = force_mag * r_unit
        force_on_2 = -force_on_1
        
        return force_on_1, force_on_2
    
    def derivatives(self, t: float, state: np.ndarray, masses: Tuple[float, float]) -> np.ndarray:
        """Calculate derivatives for ODE solver."""

        # State vector: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state
        m1, m2 = masses
        
        # Create temporary body objects
        body1 = Body(m1, [x1, y1], [vx1, vy1])
        body2 = Body(m2, [x2, y2], [vx2, vy2])
        
        # Calculate forces
        f1, f2 = self.gravitational_force(body1, body2)
        
        # Calculate accelerations
        a1 = f1 / m1
        a2 = f2 / m2
        
        # Return derivatives: [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]
        return np.array([vx1, vy1, a1[0], a1[1], vx2, vy2, a2[0], a2[1]])
    
    def calculate_energy(self, body1: Body, body2: Body) -> Tuple[float, float, float]:
        """Calculate kinetic, potential, and total energy of the system."""
        # Kinetic energy
        ke1 = 0.5 * body1.mass * np.dot(body1.velocity, body1.velocity)
        ke2 = 0.5 * body2.mass * np.dot(body2.velocity, body2.velocity)
        kinetic = ke1 + ke2
        
        # Potential energy
        r = np.linalg.norm(body2.position - body1.position)
        if r > 0:
            potential = -self.G * body1.mass * body2.mass / r
        else:
            potential = 0
        
        total = kinetic + potential
        return kinetic, potential, total
    
    def adaptive_timestep(self, body1: Body, body2: Body, base_dt: float = 0.01) -> float:
        """Calculate adaptive timestep based on system dynamics."""
        r = np.linalg.norm(body2.position - body1.position)
        v1_mag = np.linalg.norm(body1.velocity)
        v2_mag = np.linalg.norm(body2.velocity)
        
        # Scale timestep inversely with velocity and gravitational acceleration
        if r > 0:
            grav_accel = self.G * (body1.mass + body2.mass) / (r ** 2)
            char_time = min(r / max(v1_mag, v2_mag, 1e-6), 
                           np.sqrt(r / max(grav_accel, 1e-6)))
            adaptive_dt = min(base_dt, char_time * 0.01)
        else:
            adaptive_dt = base_dt * 0.1
        
        return max(adaptive_dt, 1e-6)  # Minimum timestep

class TwoBodySimulation:
    """Main simulation class integrating physics engine with visualization."""
    
    def __init__(self, body1: Body, body2: Body):
        self.body1 = body1
        self.body2 = body2
        self.physics = PhysicsEngine()
        self.time = 0.0
        self.dt = 0.01
        self.max_trail_length = 1000
        
        # Store initial conditions for energy conservation check
        self.initial_energy = self.physics.calculate_energy(body1, body2)[2]
        
    def step(self, use_adaptive_timestep: bool = True):
        """Perform one simulation step."""
        if use_adaptive_timestep:
            self.dt = self.physics.adaptive_timestep(self.body1, self.body2)
        
        # Prepare state vector for ODE solver
        state = np.array([
            self.body1.position[0], self.body1.position[1],
            self.body1.velocity[0], self.body1.velocity[1],
            self.body2.position[0], self.body2.position[1],
            self.body2.velocity[0], self.body2.velocity[1]
        ])
        
        masses = (self.body1.mass, self.body2.mass)
        
        # Solve ODE for one timestep using RK45
        sol = solve_ivp(
            lambda t, y: self.physics.derivatives(t, y, masses),
            [self.time, self.time + self.dt],
            state,
            method='RK45',
            rtol=1e-8
        )
        
        # Update body positions and velocities
        final_state = sol.y[:, -1]
        self.body1.position = final_state[0:2]
        self.body1.velocity = final_state[2:4]
        self.body2.position = final_state[4:6]
        self.body2.velocity = final_state[6:8]
        
        # Update trails
        self.body1.trail.append(self.body1.position.copy())
        self.body2.trail.append(self.body2.position.copy())
        
        # Limit trail length
        if len(self.body1.trail) > self.max_trail_length:
            self.body1.trail.pop(0)
        if len(self.body2.trail) > self.max_trail_length:
            self.body2.trail.pop(0)
        
        self.time += self.dt
        
        # Record energy for conservation check
        _, _, total_energy = self.physics.calculate_energy(self.body1, self.body2)
        self.physics.total_energy_history.append(total_energy)
        self.physics.time_history.append(self.time)
    
    def get_energy_conservation_error(self) -> float:
        """Calculate relative energy conservation error."""
        if not self.physics.total_energy_history:
            return 0.0
        
        current_energy = self.physics.total_energy_history[-1]
        return abs((current_energy - self.initial_energy) / self.initial_energy)

class MatplotlibVisualizer:
    """Matplotlib-based visualization for orbit analysis."""
    
    def __init__(self, simulation: TwoBodySimulation):
        self.sim = simulation
        self.fig, ((self.ax_orbit, self.ax_energy), 
                  (self.ax_distance, self.ax_phase)) = plt.subplots(2, 2, figsize=(12, 10))
        self.setup_plots()
    
    def setup_plots(self):
        """Initialize plot layouts."""
        # Orbit plot
        self.ax_orbit.set_title('Orbital Trajectories')
        self.ax_orbit.set_xlabel('X Position')
        self.ax_orbit.set_ylabel('Y Position')
        self.ax_orbit.set_aspect('equal')
        self.ax_orbit.grid(True, alpha=0.3)
        
        # Energy plot
        self.ax_energy.set_title('Energy Conservation')
        self.ax_energy.set_xlabel('Time')
        self.ax_energy.set_ylabel('Total Energy')
        self.ax_energy.grid(True, alpha=0.3)
        
        # Distance plot
        self.ax_distance.set_title('Inter-body Distance')
        self.ax_distance.set_xlabel('Time')
        self.ax_distance.set_ylabel('Distance')
        self.ax_distance.grid(True, alpha=0.3)
        
        # Phase space plot
        self.ax_phase.set_title('Phase Space (Body 1)')
        self.ax_phase.set_xlabel('X Position')
        self.ax_phase.set_ylabel('X Velocity')
        self.ax_phase.grid(True, alpha=0.3)
    
    def update_plots(self):
        """Update all plots with current simulation data."""
        # Clear previous plots
        for ax in [self.ax_orbit, self.ax_energy, self.ax_distance, self.ax_phase]:
            ax.clear()
        
        self.setup_plots()
        
        # Plot orbits
        if len(self.sim.body1.trail) > 1:
            trail1 = np.array(self.sim.body1.trail)
            trail2 = np.array(self.sim.body2.trail)
            
            self.ax_orbit.plot(trail1[:, 0], trail1[:, 1], 'b-', alpha=0.7, label='Body 1')
            self.ax_orbit.plot(trail2[:, 0], trail2[:, 1], 'r-', alpha=0.7, label='Body 2')
            self.ax_orbit.scatter(trail1[-1, 0], trail1[-1, 1], c='blue', s=50, zorder=5)
            self.ax_orbit.scatter(trail2[-1, 0], trail2[-1, 1], c='red', s=50, zorder=5)
            self.ax_orbit.legend()
        
        # Plot energy conservation
        if len(self.sim.physics.total_energy_history) > 1:
            self.ax_energy.plot(self.sim.physics.time_history, 
                               self.sim.physics.total_energy_history, 'g-')
            
            # Show energy conservation error
            error = self.sim.get_energy_conservation_error()
            self.ax_energy.text(0.02, 0.98, f'Rel. Error: {error:.2e}', 
                               transform=self.ax_energy.transAxes, 
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot inter-body distance
        if len(self.sim.body1.trail) > 1:
            distances = []
            for i in range(len(self.sim.body1.trail)):
                r = np.linalg.norm(np.array(self.sim.body2.trail[i]) - 
                                  np.array(self.sim.body1.trail[i]))
                distances.append(r)
            
            times = np.linspace(0, self.sim.time, len(distances))
            self.ax_distance.plot(times, distances, 'm-')
        
        # Plot phase space
        if len(self.sim.body1.trail) > 1:
            trail1 = np.array(self.sim.body1.trail)
            # Calculate velocities from position differences (approximation)
            if len(trail1) > 1:
                velocities = np.diff(trail1, axis=0) / self.sim.dt
                self.ax_phase.plot(trail1[1:, 0], velocities[:, 0], 'c-', alpha=0.7)
        
        plt.tight_layout()
        plt.draw()

class PygameVisualizer:
    """Pygame-based real-time visualization."""
    
    def __init__(self, simulation: TwoBodySimulation, width: int = 800, height: int = 600):
        pygame.init()
        self.sim = simulation
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Two Body Simulation")
        self.clock = pygame.time.Clock()
        self.scale = 1.0
        self.offset = np.array([width // 2, height // 2])
        self.running = True
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 100, 255)
        self.RED = (255, 100, 0)
        self.GREEN = (0, 255, 0)
    
    def world_to_screen(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        try:
            # Ensure world_pos is a numpy array with 2 elements
            if not isinstance(world_pos, np.ndarray):
                world_pos = np.array(world_pos, dtype=float)
            if world_pos.size != 2:
                raise ValueError("World position must have exactly 2 coordinates")
            
            # Apply scaling and offset
            screen_pos = world_pos * self.scale + self.offset
            
            # Convert to integers and ensure they are valid coordinates
            x = int(np.clip(screen_pos[0], -16383, 16383))  # Pygame has coordinate limits
            y = int(np.clip(screen_pos[1], -16383, 16383))
            
            return (x, y)
        except Exception as e:
            # Return center of screen if conversion fails
            return (self.width // 2, self.height // 2)
    
    def auto_scale(self):
        """Automatically adjust scale and offset to keep both bodies visible."""
        try:
            # Always include current positions
            all_positions = [
                self.sim.body1.position,
                self.sim.body2.position
            ]
            
            # Add trail positions if available
            if hasattr(self.sim.body1, 'trail') and len(self.sim.body1.trail) > 0:
                all_positions.extend(self.sim.body1.trail[-50:])
            if hasattr(self.sim.body2, 'trail') and len(self.sim.body2.trail) > 0:
                all_positions.extend(self.sim.body2.trail[-50:])
            
            if not all_positions:
                return
            
            # Convert to numpy array for calculations
            positions = np.array(all_positions)
            min_pos = np.min(positions, axis=0)
            max_pos = np.max(positions, axis=0)
            
            range_x = max_pos[0] - min_pos[0]
            range_y = max_pos[1] - min_pos[1]
            
            # Add padding to prevent bodies from touching screen edges
            padding = 0.2  # 20% padding
            range_x *= (1 + padding)
            range_y *= (1 + padding)
            
            if range_x > 0 and range_y > 0:
                # Calculate scale that fits the view
                scale_x = (self.width * 0.8) / range_x
                scale_y = (self.height * 0.8) / range_y
                target_scale = min(scale_x, scale_y)
                
                # Smooth scale transition
                self.scale = self.scale * 0.95 + target_scale * 0.05
                
                # Update center position
                center = (min_pos + max_pos) / 2
                target_offset = np.array([self.width // 2, self.height // 2]) - center * self.scale
                self.offset = self.offset * 0.95 + target_offset * 0.05
                
        except Exception as e:
            # If anything fails, reset to default view
            self.scale = 1.0
            self.offset = np.array([self.width // 2, self.height // 2])
    
    def draw_trail(self, trail: List[np.ndarray], color: Tuple[int, int, int], max_points: int = 200):
        """Draw orbital trail."""
        if len(trail) < 2:
            return
        
        try:
            # Get the most recent points up to max_points
            trail_points = trail[-max_points:]
            
            # Convert world coordinates to screen coordinates and ensure they are integer pairs
            screen_points = []
            for pos in trail_points:
                screen_pos = self.world_to_screen(pos)
                if (isinstance(screen_pos[0], (int, float)) and 
                    isinstance(screen_pos[1], (int, float))):
                    screen_points.append((int(screen_pos[0]), int(screen_pos[1])))
            
            # Only draw if we have at least 2 valid points
            if len(screen_points) > 1:
                pygame.draw.lines(self.screen, color, False, screen_points, 2)
        except Exception as e:
            # Skip drawing trail if there's an error
            pass
    
    def draw_body(self, body: Body, color: Tuple[int, int, int]):
        """Draw a celestial body."""
        try:
            screen_pos = self.world_to_screen(body.position)
            
            # Ensure radius is a valid number and reasonable size
            try:
                radius = float(body.radius * self.scale)
                radius = max(3.0, min(radius, 50.0))  # Clamp between 3 and 50 pixels
                radius = int(radius)
            except (TypeError, ValueError):
                radius = 5  # Default radius if calculation fails
                
            pygame.draw.circle(self.screen, color, screen_pos, radius)
            
            # Draw velocity vector with safety checks
            vel_scale = min(0.1, 50.0 / (np.linalg.norm(body.velocity) + 1e-6))  # Adaptive scale
            vel_end = body.position + body.velocity * vel_scale
            vel_screen_end = self.world_to_screen(vel_end)
            
            # Ensure coordinates are valid integers
            if (isinstance(screen_pos[0], (int, float)) and 
                isinstance(screen_pos[1], (int, float)) and
                isinstance(vel_screen_end[0], (int, float)) and
                isinstance(vel_screen_end[1], (int, float))):
                pygame.draw.line(self.screen, color, 
                               (int(screen_pos[0]), int(screen_pos[1])),
                               (int(vel_screen_end[0]), int(vel_screen_end[1])), 2)
        except Exception as e:
            # Skip drawing if there's an error
            pass
    
    def draw_info(self):
        """Draw simulation information."""
        font = pygame.font.Font(None, 36)
        
        # Time
        time_text = font.render(f"Time: {self.sim.time:.2f}", True, self.WHITE)
        self.screen.blit(time_text, (10, 10))
        
        # Energy conservation error
        error = self.sim.get_energy_conservation_error()
        error_text = font.render(f"Energy Error: {error:.2e}", True, self.WHITE)
        self.screen.blit(error_text, (10, 40))
        
        # Timestep
        dt_text = font.render(f"dt: {self.sim.dt:.4f}", True, self.WHITE)
        self.screen.blit(dt_text, (10, 70))
        
        # Distance
        distance = np.linalg.norm(self.sim.body2.position - self.sim.body1.position)
        dist_text = font.render(f"Distance: {distance:.2f}", True, self.WHITE)
        self.screen.blit(dist_text, (10, 100))
    
    def run(self, steps_per_frame: int = 1, show_matplotlib: bool = False):
        """Run the interactive simulation."""
        matplotlib_viz = None
        if show_matplotlib:
            matplotlib_viz = MatplotlibVisualizer(self.sim)
            plt.ion()
            plt.show()
        
        frame_count = 0
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Reset simulation
                        self.sim.body1.trail.clear()
                        self.sim.body2.trail.clear()
                        self.sim.physics.total_energy_history.clear()
                        self.sim.physics.time_history.clear()
                        self.sim.time = 0.0
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # Simulation steps
            for _ in range(steps_per_frame):
                self.sim.step(use_adaptive_timestep=True)
            
            # Auto-scale view
            self.auto_scale()
            
            # Clear screen
            self.screen.fill(self.BLACK)
            
            # Draw trails
            self.draw_trail(self.sim.body1.trail, self.BLUE)
            self.draw_trail(self.sim.body2.trail, self.RED)
            
            # Draw bodies
            self.draw_body(self.sim.body1, self.BLUE)
            self.draw_body(self.sim.body2, self.RED)
            
            # Draw info
            self.draw_info()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)
            
            # Update matplotlib plots periodically
            if show_matplotlib and matplotlib_viz and frame_count % 30 == 0:
                matplotlib_viz.update_plots()
                plt.pause(0.01)
            
            frame_count += 1
        
        pygame.quit()
        if show_matplotlib:
            plt.ioff()

def create_earth_moon_system():
    """Create Earth-Moon system with realistic parameters (scaled)."""
    # Scaled parameters for better visualization
    earth_mass = 5.972e24
    moon_mass = 7.342e22
    earth_moon_distance = 3.844e8
    
    # Scale factors
    distance_scale = 1e-7  # Scale distances
    velocity_scale = 1.0   # Keep velocities at real scale for better dynamics
    radius_scale = 5e-6   # Larger scale for visibility
    
    earth = Body(
        mass=earth_mass,
        position=np.array([0.0, 0.0]),
        velocity=np.array([0.0, 0.0]),
        radius=float(6.371e6 * radius_scale),  # Exaggerated for visibility
        color=(0, 100, 255)
    )
    
    # Moon orbital velocity (approximately)
    moon_orbital_velocity = 1022  # m/s
    
    moon = Body(
        mass=moon_mass,
        position=np.array([earth_moon_distance * distance_scale, 0.0]),
        velocity=np.array([0.0, moon_orbital_velocity * velocity_scale]),
        radius=float(1.737e6 * radius_scale),  # Exaggerated for visibility
        color=(200, 200, 200)
    )
    
    return earth, moon

def create_binary_star_system():
    """Create a binary star system."""
    star1 = Body(
        mass=2e30,  # Solar mass
        position=[-1e11, 0],
        velocity=[0, -1e4],
        radius=7e8,
        color=(255, 255, 0)
    )
    
    star2 = Body(
        mass=1.5e30,
        position=[1e11, 0],
        velocity=[0, 1.33e4],
        radius=6e8,
        color=(255, 100, 0)
    )
    
    return star1, star2

def analyze_orbit_properties(simulation: TwoBodySimulation, duration: float = 100):
    """Analyze orbital properties like period and eccentricity."""
    print("Analyzing orbital properties...")
    
    # Run simulation
    initial_time = simulation.time
    while simulation.time - initial_time < duration:
        simulation.step()
        
        if len(simulation.body1.trail) % 100 == 0:
            print(f"Time: {simulation.time:.2f}, Energy Error: {simulation.get_energy_conservation_error():.2e}")
    
    # Calculate orbital period (rough estimate)
    distances = []
    for i in range(len(simulation.body1.trail)):
        if i < len(simulation.body2.trail):
            r = np.linalg.norm(np.array(simulation.body2.trail[i]) - 
                              np.array(simulation.body1.trail[i]))
            distances.append(r)
    
    if distances:
        max_dist = max(distances)
        min_dist = min(distances)
        eccentricity_approx = (max_dist - min_dist) / (max_dist + min_dist)
        
        print(f"\nOrbital Analysis:")
        print(f"Maximum distance: {max_dist:.2e}")
        print(f"Minimum distance: {min_dist:.2e}")
        print(f"Approximate eccentricity: {eccentricity_approx:.4f}")
        print(f"Final energy conservation error: {simulation.get_energy_conservation_error():.2e}")

if __name__ == "__main__":
    # Example usage
    print("Two Body Planetary Simulation")
    print("Choose system:")
    print("1. Earth-Moon system")
    print("2. Binary star system")
    print("3. Custom system")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        body1, body2 = create_earth_moon_system()
        print("Created Earth-Moon system")
    elif choice == "2":
        body1, body2 = create_binary_star_system()
        print("Created binary star system")
    else:
        # Custom system
        body1 = Body(mass=1e24, position=[-1e8, 0], velocity=[0, -5e3], radius=1e6)
        body2 = Body(mass=5e23, position=[1e8, 0], velocity=[0, 1e4], radius=5e5)
        print("Created custom system")
    
    # Create simulation
    sim = TwoBodySimulation(body1, body2)
    
    print("\nChoose visualization:")
    print("1. Pygame real-time simulation")
    print("2. Matplotlib analysis")
    print("3. Both")
    print("4. Command-line analysis only")
    
    viz_choice = input("Enter choice (1-4): ").strip()
    
    if viz_choice == "1":
        pygame_viz = PygameVisualizer(sim)
        print("\nStarting Pygame simulation...")
        print("Controls: SPACE = reset, ESC = quit")
        pygame_viz.run(steps_per_frame=5)
    elif viz_choice == "2":
        matplotlib_viz = MatplotlibVisualizer(sim)
        analyze_orbit_properties(sim, duration=50)
        matplotlib_viz.update_plots()
        plt.show()
    elif viz_choice == "3":
        pygame_viz = PygameVisualizer(sim)
        print("\nStarting combined visualization...")
        print("Controls: SPACE = reset, ESC = quit")
        pygame_viz.run(steps_per_frame=5, show_matplotlib=True)
    else:
        analyze_orbit_properties(sim, duration=100)
    
    print("\nSimulation complete!")
