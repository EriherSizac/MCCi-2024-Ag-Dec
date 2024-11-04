import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

def read_data(file_path):
    """Reads the .csv file and returns a numpy array of solution vectors."""
    # Each row represents a solution vector, values are separated by spaces
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    return data.values

def non_dominated_sort(vectors):
    """Finds all non-dominated layers iteratively."""
    layers = []
    remaining_vectors = vectors.copy()

    while len(remaining_vectors) > 0:
        # Step 1: Sort lexicographically by each dimension (first by x, then by y, etc.)
        sorted_vectors = sorted(remaining_vectors, key=lambda x: tuple(x))
        
        # Step 2: Initialize the first Pareto front layer
        layer = []
        for vector in sorted_vectors:
            is_dominated = False
            for front_vector in layer:
                # Check if vector is dominated by any vector in the current layer
                if all(front_vector <= vector) and any(front_vector < vector):
                    is_dominated = True
                    break
            if not is_dominated:
                layer.append(vector)
        
        # Convert each vector in layer to a list for correct comparison
        layer_as_lists = [list(vec) for vec in layer]
        
        # Remove the current layer from remaining vectors
        remaining_vectors = np.array([vec for vec in remaining_vectors if list(vec) not in layer_as_lists])

        # Add current layer to layers list
        layers.append(np.array(layer))

    return layers

def plot_2d(layers):
    """Plots 2-D Pareto fronts with connected lines for each layer."""
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    
    for idx, layer in enumerate(layers):
        # Sort layer by the first objective for a smooth line
        layer = layer[np.argsort(layer[:, 0])]
        plt.plot(layer[:, 0], layer[:, 1], color=colors[idx], label=f'Layer {idx + 1}', marker='o')
    
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('2-D Pareto Front Layers')
    plt.legend()
    plt.show()

def plot_3d(layers):
    """Plots 3-D Pareto fronts with connected planes for each layer."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    
    for idx, layer in enumerate(layers):
        # Sort layer by the first objective for a smooth surface
        layer = layer[np.argsort(layer[:, 0])]
        ax.plot(layer[:, 0], layer[:, 1], layer[:, 2], color=colors[idx], label=f'Layer {idx + 1}', marker='o')
        
        # Create a surface using the convex hull of points in the layer (using triangulation)
        if layer.shape[0] > 2:  # Only create a surface if there are enough points
            from scipy.spatial import Delaunay
            tri = Delaunay(layer[:, :2])
            ax.plot_trisurf(layer[:, 0], layer[:, 1], layer[:, 2], triangles=tri.simplices, color=colors[idx], alpha=0.3)
    
    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')
    ax.set_zlabel('Objective 3')
    plt.title('3-D Pareto Front Layers')
    plt.legend()
    plt.show()

# Main function to execute the algorithm
def main(file_path):
    vectors = read_data(file_path)
    dimensions = vectors.shape[1]

    # Step 1: Apply the non-dominated sorting algorithm
    layers = non_dominated_sort(vectors)

    # Step 2: Plot the results based on dimensionality
    if dimensions == 2:
        plot_2d(layers)
    elif dimensions == 3:
        plot_3d(layers)
    else:
        print("This implementation only supports 2-D and 3-D problems.")

if __name__ == '__main__':
  print('Testing with problems/example1_2D.csv')
  main('./problems/example1_2D.csv')

  print('Testing with problems/example2_3D.csv')
  main('./problems/example2_3D.csv')
