import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
L = 200  # Increase grid size to 200x200 for higher resolution
nx, ny = L, L  # Grid resolution
dx = dy = 1.0  # Grid spacing
dt = 5  # Larger time step to make the fluid evolve faster
Re = 5000  # Increased Reynolds number for faster dynamics
U = 1.0  # Characteristic velocity (arbitrary units)
L_scale = 100  # Characteristic length scale (in grid units)

# Calculate kinematic viscosity from Reynolds number
nu = U * L_scale / Re  # Kinematic viscosity

# Density and conductivity (arbitrary values)
rho = 1.0
sigma = 1.0

# Magnetic field (static, uniform) - Not plotted but will affect fluid dynamics
Bx = 1.0
By = 0.0

# Initialize velocity field (circular initial condition: vortex)
def initialize_velocity(nx, ny, center_x, center_y, strength=1.0):
    """
    Initializes the velocity field as a circular vortex around the center of the domain.
    """
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)  # Radial distance from the center
    
    # Avoid division by zero at the center
    r[r == 0] = 1e-10  # Small value to avoid division by zero
    
    # Tangential velocity in polar coordinates (v_theta = C / r)
    v_theta = strength / r  # Decaying velocity with radius
    
    # Calculate velocity components (vx, vy) from the tangential velocity
    vx = -v_theta * (y - center_y) / r
    vy = v_theta * (x - center_x) / r
    
    return vx, vy

# External rotation (constant)
omega = 0.1  # Rotation rate (angular velocity)

# Initial conditions: vortex at the center
center_x, center_y = nx // 2, ny // 2
vx, vy = initialize_velocity(nx, ny, center_x, center_y)

# Pressure field (initially zero)
pressure = np.zeros((ny, nx))

# Function to compute the Lorentz force
def lorentz_force(vx, vy, Bx, By, sigma):
    Jx = sigma * (np.gradient(vx, axis=1) + np.gradient(vy, axis=0))  # Simplified current density
    Jy = -sigma * (np.gradient(vx, axis=0) + np.gradient(vy, axis=1))
    fx = Jx * By - Jy * Bx  # Lorentz force x-component
    fy = Jy * Bx - Jx * By  # Lorentz force y-component
    return fx, fy

# Navier-Stokes solver with Lorentz force
def step(vx, vy, dt, nu, rho, Bx, By, sigma, dx, dy):
    fx, fy = lorentz_force(vx, vy, Bx, By, sigma)
    
    # Update velocity field using simplified Navier-Stokes equations
    vx[1:-1, 1:-1] += dt * (-vx[1:-1, 1:-1] * np.gradient(vx, axis=1)[1:-1, 1:-1] 
                             - vy[1:-1, 1:-1] * np.gradient(vx, axis=0)[1:-1, 1:-1] 
                             - np.gradient(pressure, axis=1)[1:-1, 1:-1] / rho
                             + nu * (np.gradient(np.gradient(vx, axis=1), axis=1)[1:-1, 1:-1] 
                                     + np.gradient(np.gradient(vx, axis=0), axis=0)[1:-1, 1:-1])
                             + fx[1:-1, 1:-1])
    
    vy[1:-1, 1:-1] += dt * (-vx[1:-1, 1:-1] * np.gradient(vy, axis=1)[1:-1, 1:-1] 
                             - vy[1:-1, 1:-1] * np.gradient(vy, axis=0)[1:-1, 1:-1] 
                             - np.gradient(pressure, axis=0)[1:-1, 1:-1] / rho
                             + nu * (np.gradient(np.gradient(vy, axis=1), axis=1)[1:-1, 1:-1] 
                                     + np.gradient(np.gradient(vy, axis=0), axis=0)[1:-1, 1:-1])
                             + fy[1:-1, 1:-1])
    
    return vx, vy

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Animation function
def update(frame):
    global vx, vy
    vx, vy = step(vx, vy, dt, nu, rho, Bx, By, sigma, dx, dy)
    
    ax.clear()
    # Use streamplot to show fluid motion
    ax.streamplot(np.arange(nx), np.arange(ny), vx, vy, color=np.sqrt(vx**2 + vy**2), linewidth=1, cmap='coolwarm', density=2)
    
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_title(f"Magnetic Fluid Flow (Re = {Re}, Frame {frame})")
    return []

# Create animation with faster fluid dynamics (larger dt)
ani = FuncAnimation(fig, update, frames=200, interval=30, blit=False)
plt.show()
