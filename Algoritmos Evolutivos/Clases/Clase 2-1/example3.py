"""
Example 3: Contour plot of an NLP.

f(x, y) = x^2 + y^2
such that
x, y in [-1, 1]
g(x, y) = x - y <= 0
h(x, y) = x^2 + y^2 - 0.5
""" 
import numpy as np
import matplotlib.pyplot as plt

# Define the grid for the plot
x = np.linspace(-1, 1, 400)
y = np.linspace(-1, 1, 400)
X, Y = np.meshgrid(x, y)

# Define the function f(x, y)
Z = X**2 + Y**2

# Define the constraint g(x, y)
constraint = X - Y <= 0

# Create the contour plot
plt.figure(figsize=(8, 8))

# Plot contours of f(x, y) only in the feasible region
contour_f = plt.contourf(X, Y, Z, levels=np.linspace(0, 2, 30), cmap='viridis', alpha=0.5)

# Overlay the boundary of the constraint g(x, y)
plt.plot(x, x, 'r-', label=r'$x - y = 0$', linewidth=2)

# Plot the equality constraint h(x, y) = 0.5
plt.contour(X, Y, Z, levels=[0.5], colors='blue', linewidths=2, linestyles='--', label='h(x, y) = 0.5')

# Add labels and title
plt.title(r'Contour plot of $f(x, y) = x^2 + y^2$ with $x - y \leq 0$ and $h(x, y) = 0.5$')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.axis('equal')
plt.xlim(-1, 1)
plt.ylim(-1, 1)

# Add a legend for clarity
plt.legend(['$x - y = 0$', r'$h(x, y) = 0.5$'])

# Show the color bar for the contour plot of f(x, y)
plt.colorbar(contour_f, label=r'$f(x, y) = x^2 + y^2$')

# Show the plot
plt.show()

