"""
Example 1: Contour plot of an NLP.

f(x, y) = abs(x*y)
such that
x, y in [-1, 1]
h(x, y) = x^2+ y ^2 = 1
""" 
import numpy as np
import matplotlib.pyplot as plt

# Define the grid for the plot
x = np.linspace(-1, 1, 400)
y = np.linspace(-1, 1, 400)
X, Y = np.meshgrid(x, y)

# Define the function f(x, y)
Z = np.abs(X * Y)

# Create the contour plot
plt.figure(figsize=(8, 8))

# Plot the contour of f(x, y)
contour = plt.contourf(X, Y, Z, levels=np.linspace(0, 1, 30), cmap='viridis', alpha=0.7)

# Overlay the equality constraint g(x, y) = 1 (the unit circle)
circle = plt.contour(X, Y, X**2 + Y**2, levels=[1], colors='blue', linewidths=2, linestyles='--')

# Add labels and title
plt.title(r'Contour plot of $f(x, y) = |xy|$ with $g(x, y) = x^2 + y^2 = 1$')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.axis('equal')
plt.xlim(-1, 1)
plt.ylim(-1, 1)

# Show the color bar for the contour plot of f(x, y)
plt.colorbar(contour, label=r'$f(x, y) = |xy|$')

# Show the plot
plt.legend(['$g(x, y) = x^2 + y^2 = 1$'])
plt.show()

