from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Hill Climber Method with overflow control
def hill_climber(f, x0, step_size=0.01, max_iter=1000):
    x = np.array(x0)
    best_val = f(*x)
    
    for i in tqdm(range(max_iter), desc="Hill Climber Progress"):
        # Generate a random perturbation
        perturbation = np.random.uniform(-step_size, step_size, size=x.shape)
        new_x = x + perturbation
        new_val = f(*new_x)
        
        # If the new point is better, move to it
        if new_val < best_val:
            x = new_x
            best_val = new_val
        
        # Clip the values to avoid overflow
        x = np.clip(x, -1e2, 1e2)
        
        # Print the progress every 100 iterations
        if i % 100 == 0 or i == max_iter - 1:
            print(f"Iteration {i}: Best Value = {best_val}, Current Point = {x}")
    
    return x, best_val

# Gradient Descent with Wolfe Conditions and overflow control
def gradient_descent_wolfe(f, grad_f, x0, t=1.0, c1=1e-4, c2=0.9, max_iter=100):
    x = np.array(x0)
    
    def wolfe_conditions(t, grad, direction):
        return (f(*(x + t * direction)) <= f(*x) + c1 * t * np.dot(grad, direction)) and \
               (np.dot(grad_f(*(x + t * direction)), direction) >= c2 * np.dot(grad, direction))
    
    for i in tqdm(range(max_iter), desc="Gradient Descent Progress"):
        grad = grad_f(*x)
        direction = -grad
        
        # Step size control using Wolfe conditions
        while not wolfe_conditions(t, grad, direction):
            if f(*(x + t * direction)) > f(*x) + c1 * t * np.dot(grad, direction):
                t *= 0.5
            elif np.dot(grad_f(*(x + t * direction)), direction) < c2 * np.dot(grad, direction):
                t *= 2.0
            else:
                break
        
        x = x + t * direction
        
        # Clip the values to avoid overflow
        x = np.clip(x, -1e2, 1e2)
        
        # Print the progress
        if i % 10 == 0 or i == max_iter - 1:
            print(f"Iteration {i}: Current Value = {f(*x)}, Current Point = {x}")
    
    return x, f(*x)

# Newton's Method with overflow control
def newton_method(f, grad_f, hess_f, x0, max_iter=100):
    x = np.array(x0)
    
    for i in tqdm(range(max_iter), desc="Newton Method Progress"):
        grad = grad_f(*x)
        hess = hess_f(*x)
        
        # Solve for the search direction
        direction = np.linalg.solve(hess, -grad)
        
        x = x + direction
        
        # Clip the values to avoid overflow
        x = np.clip(x, -1e2, 1e2)
        
        # Print the progress
        if i % 10 == 0 or i == max_iter - 1:
            print(f"Iteration {i}: Current Value = {f(*x)}, Current Point = {x}")
        
        if np.linalg.norm(grad) < 1e-8:
            break
    
    return x, f(*x)

# Define the functions
def func1(x1, x2):
    return np.clip(-2*x1**2 + 3*x1*x2 - 1.5*x2**2 - 1.3, -1e10, 1e10)

def func2(x1, x2):
    return np.clip((4 - 2.1*x1**2 + x1**4 / 3) * x1**2 + x1*x2 + (-4 + 4*x2**2) * x2**2, -1e10, 1e10)

# Define the gradients
def grad_func1(x1, x2):
    df_dx1 = -4*x1 + 3*x2
    df_dx2 = 3*x1 - 3*x2
    return np.array([df_dx1, df_dx2])

def grad_func2(x1, x2):
    df_dx1 = (8*x1 - 4.2*x1**3 + 4*x1**3) + x2
    df_dx2 = x1 + (8*x2**3 - 8*x2)
    return np.array([df_dx1, df_dx2])

# Define the Hessians
def hess_func1(x1, x2):
    d2f_dx1x1 = -4
    d2f_dx1x2 = 3
    d2f_dx2x1 = 3
    d2f_dx2x2 = -3
    return np.array([[d2f_dx1x1, d2f_dx1x2], [d2f_dx2x1, d2f_dx2x2]])

def hess_func2(x1, x2):
    d2f_dx1x1 = 8 - 12.6*x1**2 + 12*x1**2
    d2f_dx1x2 = 1
    d2f_dx2x1 = 1
    d2f_dx2x2 = 24*x2**2 - 8
    return np.array([[d2f_dx1x1, d2f_dx1x2], [d2f_dx2x1, d2f_dx2x2]])

# Starting points
x0_func1 = [-4, 4]
x0_func2 = [0.5, 1]

# Test Hill Climber
print("Testing Hill Climber Method:")
hc_result_func1 = hill_climber(func1, x0_func1)
hc_result_func2 = hill_climber(func2, x0_func2)

# Test Newton's Method
print("Testing Newton's Method:")
newton_result_func1 = newton_method(func1, grad_func1, hess_func1, x0_func1)
newton_result_func2 = newton_method(func2, grad_func2, hess_func2, x0_func2)

# Test Gradient Descent with Wolfe Conditions
print("Testing Gradient Descent with Wolfe Conditions:")
gd_result_func1 = gradient_descent_wolfe(func1, grad_func1, x0_func1)
gd_result_func2 = gradient_descent_wolfe(func2, grad_func2, x0_func2)



# Print results
print("\nResults:")
print("Hill Climber Result Function 1:", hc_result_func1)
print("Hill Climber Result Function 2:", hc_result_func2)

print("Gradient Descent Wolfe Result Function 1:", gd_result_func1)
print("Gradient Descent Wolfe Result Function 2:", gd_result_func2)

print("Newton Method Result Function 1:", newton_result_func1)
print("Newton Method Result Function 2:", newton_result_func2)

# Visualization for Function 1
x1_vals = np.linspace(-5, 5, 400)
x2_vals = np.linspace(-5, 5, 400)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z1 = func1(X1, X2)

plt.contour(X1, X2, Z1, levels=50)
plt.scatter(hc_result_func1[0][0], hc_result_func1[0][1], color='red', label='Hill Climber')
plt.scatter(gd_result_func1[0][0], gd_result_func1[0][1], color='blue', label='Gradient Descent Wolfe')
plt.scatter(newton_result_func1[0][0], newton_result_func1[0][1], color='green', label='Newton')
plt.legend()
plt.title('Optimization Paths for Function 1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
