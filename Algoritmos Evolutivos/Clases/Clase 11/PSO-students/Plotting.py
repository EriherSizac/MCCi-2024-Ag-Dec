import matplotlib.pyplot as plt
import numpy as np


class Plot:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.line = None
        self.annot = None
        self.z_labels = None

    def analysis_plot(self, obj_function, best_individuals, execution):
        # Preparing data
        xmin = obj_function.get_xmin()
        xmax = obj_function.get_xmax()
        x = np.linspace(xmin[0], xmax[0], 1000)
        y = np.linspace(xmin[1], xmax[1], 1000)
        Z = np.empty((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                aux = np.array([x[i], y[j]])
                Z[i, j]  = obj_function.evaluate(aux)
        X, Y = np.meshgrid(x, y)
        # Initialize figure   
        plt.rcParams.update({'font.size': 12})
        
        self.fig = plt.figure()
        self.fig.suptitle("{0} - Execution {1}".format(obj_function.get_name(), execution))
        # Plot 3D contour
        axis1 = self.fig.add_subplot(1,3,1, projection='3d')
        axis1.plot_surface(X, Y, Z, cmap='plasma')
        axis1.set_xlabel("$x_1$")
        axis1.set_ylabel("$x_2$")
        axis1.set_zlabel("$f(x_1, x_2)$")
        axis1.title.set_text('Fitness landscape')
        # Plot 2D contour  
        self.ax = self.fig.add_subplot(1,3,2)
        self.ax.contourf(X, Y, Z, 20, cmap='plasma')
        self.ax.set_xlabel("$x_1$")
        self.ax.set_ylabel("$x_2$")  
        self.ax.title.set_text('Contour plot and sequence of points')
        self.annot = self.ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

        x1 = np.empty(len(best_individuals))
        x2 = np.empty(len(best_individuals))
        self.z_labels = np.empty(len(best_individuals))
        # Add best invidual from each generation to the contour plot
        for i in range(len(best_individuals)):
            decision_vector = best_individuals[i].get_x()
            x1[i] = decision_vector[0]
            x2[i] = decision_vector[1]
            self.z_labels[i] = best_individuals[i].get_objective_value()            
        self.line, = plt.plot(x1, x2, marker='o', color="white")
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        # Convergence plot
        axis3 = self.fig.add_subplot(1,3,3)
        axis3.plot(np.arange(1, len(best_individuals) + 1, 1), self.z_labels)
        axis3.scatter(np.arange(1, len(best_individuals) + 1, 1), self.z_labels)
        axis3.title.set_text("Convergence plot")
        axis3.set_xlabel("Generation")
        axis3.set_ylabel("Objective value")
        axis3.set_xscale('log')
        plt.subplots_adjust(wspace=0.4)
        plt.show()


    def update_annot(self, ind):
        x,y = self.line.get_data()
        self.annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
        text = "$f(t={}) = {}$".format(ind["ind"][0], self.z_labels[ind["ind"][0]])
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_alpha(0.4)


    def hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.line.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()