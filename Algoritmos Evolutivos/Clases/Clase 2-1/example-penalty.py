import numpy as np
import matplotlib.pyplot as plt
import sys

def g(x, y, A, B):
    return (x**2 + y**2 + 
            A * np.maximum(0, x - y)**2 + 
            B * np.abs(x**2 + y**2 - 0.5)**2)

def f(x, y):
    return x**2 + y**2

# Check if the user has provided A and B as command line arguments
if len(sys.argv) != 3:
    print("Usage: python script.py <A> <B>")
    sys.exit(1)

# Read A and B from command line
A = float(sys.argv[1])
B = float(sys.argv[2])

# Create a grid of points
x = np.linspace(-1, 1, 400)  # x in [-1, 1]
y = np.linspace(-1, 1, 400)  # y in [-1, 1]
X, Y = np.meshgrid(x, y)

# Compute g(x, y) and f(x, y)
Z_g = g(X, Y, A, B)
Z_f = f(X, Y)

# Create the plots
fig = plt.figure(figsize=(12, 8))

# 3D Surface Plot for g(x, y)
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z_g, cmap='viridis', alpha=0.7)
ax1.set_title('3D Surface Plot of g(x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('g(x, y)')

# Contour Plot for g(x, y)
contour_g = ax1.contour(X, Y, Z_g, levels=20, zdir='z', offset=ax1.get_zlim()[0], cmap='viridis', alpha=0.5)

# 3D Surface Plot for f(x, y)
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z_f, cmap='plasma', alpha=0.7)
ax2.set_title('3D Surface Plot of f(x, y)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x, y)')

# Contour Plot for f(x, y)
contour_f = ax2.contour(X, Y, Z_f, levels=20, zdir='z', offset=ax2.get_zlim()[0], cmap='plasma', alpha=0.5)

# Show plots
plt.tight_layout()
plt.show()

