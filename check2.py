import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

L = 1  
n = 20  
dx = L / n  
mu = 1  
density = 1  
voltage = 1  
magnetic_centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]  

x = np.linspace(0, L, n)
y = np.linspace(0, L, n)
X, Y = np.meshgrid(x, y)

def magnetic_field(x, y, centers):
    Bx, By = np.zeros_like(x), np.zeros_like(y)
    for cx, cy in centers:
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        angle = np.arctan2(y - cy, x - cx)
        Bx += np.cos(angle) / r**2
        By += np.sin(angle) / r**2
    return Bx, By

Bx, By = magnetic_field(X, Y, magnetic_centers)

def fluid_dynamics(t, u_flat):
    u = u_flat.reshape(2, n, n)  
    u_x, u_y = u  

    du_x_dt = -(1 / mu) * (np.gradient(u_x, axis=1) + np.gradient(u_x, axis=0))  
    du_y_dt = -(1 / mu) * (np.gradient(u_y, axis=1) + np.gradient(u_y, axis=0))  
    
    du_x_dt += voltage * By  
    du_y_dt -= voltage * Bx  
    return np.concatenate([du_x_dt.flatten(), du_y_dt.flatten()])

initial_conditions = np.zeros((2, n, n))  

t_span = (0, 10)
t_eval = np.linspace(*t_span, 100)

solution = solve_ivp(fluid_dynamics, t_span, initial_conditions.flatten(), t_eval=t_eval)

def plot_fluid_flow_contour(solution, X, Y):
    plt.figure(figsize=(8, 6))
    
    for i in range(0, solution.t.shape[0], 10):  
        u = solution.y[:, i].reshape(2, n, n)  
        u_x, u_y = u  

        velocity_magnitude = np.sqrt(u_x**2 + u_y**2)

        levels = np.linspace(velocity_magnitude.min(), velocity_magnitude.max(), 10)

        if np.any(np.diff(levels) <= 0):
            print("Warning: Contour levels are not strictly increasing.")
            levels = np.linspace(velocity_magnitude.min(), velocity_magnitude.max(), 10)

        contour = plt.contourf(X, Y, velocity_magnitude, levels=levels, cmap='viridis')

    plt.title("Fluid Flow under Magnetic Field and Voltage (Contour Plot)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(contour, label="Velocity Magnitude")
    plt.show()

plot_fluid_flow_contour(solution, X, Y)
