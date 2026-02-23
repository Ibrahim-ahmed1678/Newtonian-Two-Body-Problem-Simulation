"""
N-body Orbit Simulator (Velocity Verlet) with Pygame + optional Matplotlib live plots.

Controls:
  Space  - pause / resume
  + / =  - double simulation speed (time scale)
  - / _  - halve simulation speed
  r      - reset to initial conditions
  t      - toggle trails
  m      - toggle Matplotlib live plot (two panels: trajectories & energy error)
  Mouse wheel - zoom (about mouse)
  Left-drag    - pan camera
  Esc or window close - quit

Requirements:
  pip install pygame numpy
  matplotlib is optional (only needed if you press 'm').
"""

import pygame
import sys
from Classes.NBodySimulation import NBodySimulation
from Classes.LivePlot import LivePlot
from Classes.Camera import Camera
from support import ( 
    create_two_body_star_planet, 
    create_five_body_demo, 
    WIDTH, 
    HEIGHT, 
    BACKGROUND_COLOR, 
    INITIAL_TIME_SCALE, 
    PHYSICS_DT, 
    MAX_SUBSTEPS, 
    TRAIL_MAX_LEN 
    )



# ------------------ Main Application ------------------
def main():
    # choose preset — you can expand this list or create body lists interactively
    presets = {
        1: ("Two-body star+planet", create_two_body_star_planet),
        2: ("Five-body demo", create_five_body_demo),
        3: ("Two-body star+star", create_two_body_star_planet)
    }
    print("\n\n==============================")
    print("N-body Orbit Simulator")
    print("==============================\n")
    print("Choose a preset system:\n")
    print("1: Two-body star+planet")
    print("2: Five-body demo")
    print("3: Two body star-star\n")

    chosen = int(input("Enter choice: ").strip())  # change to 2 to start with 5-body, or keep choice interactive if desired

    # create bodies according to preset; some presets may support extra options
    if chosen == 2:
        # ask user whether to ignore planet-planet gravity
        bodies = presets[chosen][1]()
        sim = NBodySimulation(bodies, central_body_index=0)
    else:
        bodies = presets[chosen][1](chosen)
        sim = NBodySimulation(bodies)
    
    cam = Camera(WIDTH, HEIGHT, scale=220.0)
    liveplot = LivePlot()

    # state
    trails_on = True
    paused = False
    time_scale = INITIAL_TIME_SCALE
    sim_time = 0.0
    last_real_time = pygame.time.get_ticks() / 1000.0

    # initial energy snapshot for relative error baseline
    E0 = sim.total_energy()
    sim.energy_history.append(E0)
    sim.time_history.append(sim.time)

    # store a snapshot as initial state for reset
    sim.initial_state = sim._snapshot_state()

    # Pygame init
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("N-body Orbit Simulator")
    clock = pygame.time.Clock()
    hud_font = pygame.font.SysFont("Consolas", 16)

    running = True

    while running:

        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    time_scale *= 2.0

                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    time_scale = max(1e-8, time_scale * 0.5)

                elif event.key == pygame.K_r:
                    sim.reset()
                    
                    # recompute baseline energy
                    E0 = sim.total_energy()
                    sim.energy_history = [E0]
                    sim.time_history = [sim.time]
                    sim_time = 0.0

                elif event.key == pygame.K_t:
                    trails_on = not trails_on
                    if not trails_on:
                        for b in sim.bodies:
                            b.trail.clear()

                elif event.key == pygame.K_m:
                    liveplot.toggle()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    cam.start_drag(*event.pos)

                elif event.button == 4:  # scroll up -> zoom in
                    mx, my = event.pos
                    world_before = cam.screen_to_world(mx, my)
                    cam.scale *= 1.2
                    world_after = cam.screen_to_world(mx, my)
                    cam.offset += (world_after - world_before)

                elif event.button == 5:  # scroll down -> zoom out
                    mx, my = event.pos
                    world_before = cam.screen_to_world(mx, my)
                    cam.scale /= 1.2
                    world_after = cam.screen_to_world(mx, my)
                    cam.offset += (world_after - world_before)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    cam.end_drag()

            elif event.type == pygame.MOUSEMOTION:
                if cam.dragging:
                    cam.drag(*event.pos)

        # time management
        current_real_time = pygame.time.get_ticks() / 1000.0
        real_dt = current_real_time - last_real_time
        last_real_time = current_real_time
        target_sim_dt = 0.0 if paused else real_dt * time_scale

        # decide substeps (keep each substep reasonably close to PHYSICS_DT)
        if target_sim_dt <= 0:
            substeps = 0
            dt_per_step = 0.0

        else:
            # try to use integer count near target/PHSICS_DT
            ideal_steps = int(max(1, round(target_sim_dt / PHYSICS_DT)))
            substeps = min(MAX_SUBSTEPS, max(1, ideal_steps))
            dt_per_step = target_sim_dt / substeps

        # integrate
        for _ in range(substeps):
            sim.velocity_verlet_step(dt_per_step)
            sim_time += dt_per_step

            # update trails
            if trails_on:
                for b in sim.bodies:
                    b.trail.append(b.pos.copy())

                    if len(b.trail) > TRAIL_MAX_LEN:
                        b.trail.pop(0)

            # record energy/time history occasionally (every frame step is okay)
            sim.energy_history.append(sim.total_energy())
            sim.time_history.append(sim.time)

        # Draw
        screen.fill(BACKGROUND_COLOR)

        # draw trails
        if trails_on:
            for b in sim.bodies:
                if len(b.trail) >= 2:
                    pts = [cam.world_to_screen(p) for p in b.trail]

                    # draw with fading alpha by grouping segments (pygame doesn't support per-vertex alpha easily)
                    pygame.draw.lines(screen, b.color, False, pts, 2)

        # draw bodies
        for b in sim.bodies:
            sx, sy = cam.world_to_screen(b.pos)
            # scale radius to be visible but not huge
            screen_radius = int(max(2, b.radius * cam.scale / 100.0))
            pygame.draw.circle(screen, b.color, (sx, sy), screen_radius)

        # HUD (simple)
        E = sim.total_energy()
        energy_err = (E - sim.energy_history[0]) / abs(sim.energy_history[0]) if abs(sim.energy_history[0]) > 0 else 0.0
        hud_lines = [
            f"Bodies: {len(sim.bodies)}",
            f"Sim time: {sim.time:.6f}",
            f"Time scale: {time_scale:.6g} (press +/-)",
            f"Physics dt (approx): {PHYSICS_DT:.1e} | substeps/frame: {substeps}",
            f"Energy rel change: {energy_err:.3e}  (press r to reset baseline)",
            f"Paused: {paused}  Trails: {'On' if trails_on else 'Off'} (t)  Plots: {'On' if liveplot.enabled else 'Off'} (m)"
        ]

        y = 8
        for line in hud_lines:
            surf = hud_font.render(line, True, (220, 220, 220))
            screen.blit(surf, (8, y))
            y += surf.get_height() + 6

        pygame.display.flip()

        # update matplotlib if enabled (non-blocking)
        if liveplot.enabled:
            try:
                liveplot.update(sim, cam)

            except Exception as e:
                # If plotting fails, disable to avoid freezing
                print("Live plot error, disabling plotting:", e)
                liveplot.toggle()

        clock.tick(60)  # cap FPS

    pygame.quit()

    # close matplotlib figure if open
    if liveplot.enabled and liveplot.plt is not None:
        try:
            liveplot.plt.close(liveplot.fig)

        except Exception:
            pass

    sys.exit()

if __name__ == "__main__":
    main()
