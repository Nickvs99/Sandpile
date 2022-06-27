import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.optimize import curve_fit
from scipy.stats import mode
from matplotlib import colors as mcolors
from collections import Counter

from plots import plot_time_series, plot_size_probability

def fit_func(s, tau, a, C):
    """ A powerlaw distribution with exponential falloff. C is a normalization constant. """

    return C * s**(-tau) * a**s

def scaled_fit_func(x, tau, a, C):

    return np.log(fit_func(x, tau, a, C))

class Sandpile_model:

    def __init__(self, grid_size=100, n_grain_types=2, grain_odds = [], crit_values=[1, 2], n_steps=10000, boundary_con=False, add_method="position", init_method="random", seed=None):
        """
        The extended sandpile model. This model supports multiple types of grains.

        Arguments:
         - grid_size (int): size of the grid
         - n_grain_types (int): the number of types of grains
         - crit_valeus (list): The critical values of the grains. This list needs to be the same length
                as there are n_grain_types.
         - n_steps (int): The number of steps the model runs.
         - boundary_con (bool): True if walled model (sand is not removed) if False sand is removed from the sides
         - add_method (string): The method at which new grains are added
         - init_method (string): The method specifies how the initial grid is setup
         - seed (int): Set the seed for the random number generators. If None then a random seed is used.
        """

        # Check for valid input parameters
        if len(crit_values) != n_grain_types:
            raise Exception("Error: length of crit values does not match with the number of grain types")
            
        if grain_odds == []:
            grain_odds = [1/n_grain_types] * n_grain_types
            
        if len(grain_odds) != n_grain_types:
            raise Exception("Error: length of grain odds does not match with the number of grain types")
            
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.grid_size = grid_size
        self.n_grain_types = n_grain_types
        self.crit_values = crit_values
        self.n_steps = n_steps
        self.boundary_con = boundary_con
        self.add_method = add_method
        self.init_method = init_method
        self.grain_odds = grain_odds
        
        self.initialize_grids()

        self.current_step = 0
        self.data = np.zeros(self.n_steps)

    def initialize_grids(self):
        """
        Initalizes both the height_grid and grid_3D.
        height_grid stores the height at each location i,j
        grid_3D stores the grains at each location i,j

        Possible methods:
         - None: Create an empty grid
         - random: create a random grid with z_i <= z^th for all i
        """

        # Create an empty grid with zero height
        self.height_grid = np.zeros((self.grid_size, self.grid_size))
        
        self.grid_3D = []
        for i in range(self.grid_size):
            self.grid_3D.append([])
            
            for j in range(self.grid_size):
                self.grid_3D[i].append([])

        # Prepare the system in an arbitrary stable configuration with z_i <= z^th for all i.
        if self.init_method == "random":
            
            min_treshold = min(self.crit_values)

            for i in range(self.grid_size):
                for j in range(self.grid_size):

                    n_placements = np.random.randint(min_treshold + 1)

                    for k in range(n_placements):
                        self.add_grain([i, j])

    def run(self):

        while self.current_step < self.n_steps:
            self.update()

            if self.current_step % 1000 == 0:
                print(f"Step {self.current_step} of {self.n_steps}", end="\r")

    def avalanche_step(self, i, j):
        # Check if a grain is present at location i,j
        if not self.grid_3D[i][j]:
            return None
        
        # Get neighbour heights and whether they exist
        neighbour_heights, is_inside = self.get_neighbours(self.height_grid, i, j)
        pile_fall = (sum(is_inside) if self.boundary_con else 4)
        grain_type = self.grid_3D[i][j][-1]

        # Check whether an avalanche should occur
        if self.height_grid[i][j] - min(neighbour_heights) >= self.crit_values[grain_type] and self.height_grid[i][j] >= pile_fall:
            
            top_grains = [self.grid_3D[i][j].pop() for _ in range(pile_fall)]
            self.height_grid[i][j] -= pile_fall
            random.shuffle(top_grains)

            next_coords = [(i, j)]
            itter = 0
            for i2, j2 in ((i-1,j), (i,j-1), (i+1,j), (i,j+1)):
                if is_inside[itter]:
                    self.height_grid[i2][j2] += 1
                    self.grid_3D[i2][j2].append(top_grains.pop())
                    next_coords.append((i2, j2))
                itter += 1
    
            return next_coords

        return None

    def update(self):
        
        new_grain_pos = self.get_position()   
        self.add_grain(new_grain_pos)

        # update the model simultaneously
        avalanche_stack = [new_grain_pos]
        self.data[self.current_step] = 0

        while avalanche_stack:
            next_pos = avalanche_stack.pop()
            new_step = self.avalanche_step(next_pos[0], next_pos[1])
            if new_step is not None:
                self.data[self.current_step] += 1

                avalanche_stack.extend(new_step)

        self.current_step += 1

    def add_grain(self, position):

        grain_type = self.get_random_grain_type()
        self.grid_3D[position[0]][position[1]].append(grain_type)
        self.height_grid[position[0]][position[1]] += 1

        return position

    def get_random_grain_type(self):
        return np.random.choice(np.arange(0, self.n_grain_types), p = self.grain_odds)

    def get_position(self, cords=[-1,-1], std=-1):
        """
        add value to matrix using various methods:
         - position: at given postion standard is center
         - random: at random position
         - normal: at a normal distribution with given stanardard deviation around given position
                   standard postion is center and standard deviation is 1/6 th of the matrix size (3 SD intervall covers 99.9% of the matrix)
        """

        if self.add_method.lower() == "random":
            randcords = [np.random.randint(self.grid_size - 1), np.random.randint(self.grid_size - 1)]
            return randcords
        
        # set values of cordinates if no values are given
        if cords[0] == -1:
            cords[0] = math.floor(self.grid_size/2)
        if cords[1] == -1:
            cords[1] = math.floor(self.grid_size/2)
        
        if self.add_method.lower() == "position":
            return cords
        
        if self.add_method.lower() == "normal":
            # set value of standard deviation
            if std == -1:
                std = math.ceil(self.grid_size/6)
            
            randcords = [max(min(int(np.round(np.random.normal(cords[0], std))), size - 1),0), max(min(int(np.round(np.random.normal(cords[1], std))), size - 1),0)]

            return randcords

    def get_neighbours(self, matrix, i, j):
        """
        find the neighbour values of a given cell
        Returns the values and gives the neighbours that exist given the dataset
        in order: left, upper, right and down neighbouring cells.
        """
        
        if self.boundary_con:
            neighbours = [float('inf'), float('inf'), float('inf'), float('inf')]
        else:
            neighbours = [0,0,0,0]
        real = [0,0,0,0]

        if i >= 1:
            neighbours[0] = matrix[i - 1][j]
            real[0] = 1
        if j >= 1:
            neighbours[1] = matrix[i][j - 1]
            real[1] = 1

        if i < self.grid_size - 1:
            neighbours[2] = matrix[i + 1][j]
            real[2] = 1
        if j < self.grid_size - 1:
            neighbours[3] = matrix[i][j + 1]
            real[3] = 1
            
        return neighbours, real
        
    def calc_fit_parameters(self, xdata, ydata):

        # Remove data entries where the ydata is zero
        xdata = xdata[ydata != 0]
        ydata = ydata[ydata != 0]

        # Fit to the scaled function to minimize the relative error instead of the absolute error
        popt, pcov = curve_fit(scaled_fit_func, xdata, np.log(ydata))

        return popt, pcov

    def collect_time_series_data(self):
        
        ts = np.arange(self.n_steps)
        
        avalanche_sizes = np.array(self.data)
        max_avalanche_size = np.amax(avalanche_sizes)

        # Normalize the avalanche sizes
        return ts, avalanche_sizes / max_avalanche_size

    def collect_size_probability_data(self, n_bins=20):

        avalanche_sizes = np.array(self.data)
        avalanche_sizes = avalanche_sizes[avalanche_sizes != 0]

        min_avalanche_size = np.amin(avalanche_sizes)
        max_avalanche_size = np.amax(avalanche_sizes)

        # Since the x scale is logaritmic the bins also have to be logaritmic
        bins = np.unique(np.logspace(np.floor(np.log10(min_avalanche_size)), np.ceil(np.log10(max_avalanche_size)), num=n_bins, dtype=int))
        hist, bin_edges = np.histogram(avalanche_sizes, bins=bins, density=True)

        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

        return np.array(bin_centers), np.array(hist)

    def plot_time_series(self):

        ts, time_series_data = self.collect_time_series_data()

        plot_time_series([(ts, time_series_data)])

    def plot_size_probability(self, n_bins=20):

        bin_centers, hist = self.collect_size_probability_data(n_bins=n_bins)
        
        # Remove data entries where the data is zero, this removes the vertical line
        # And smoothens the data at very low probabilities due to missing avalanches
        bin_centers = bin_centers[hist != 0]
        hist = hist[hist != 0]

        # Plot fit
        popt, pcov = self.calc_fit_parameters(bin_centers, hist)   
        plt.plot(bin_centers, fit_func(bin_centers, *popt), label=f"fit: tau={popt[0]:.3f}, a={popt[1]:.3f}")
        
        plot_size_probability([(bin_centers, hist)], labels=["raw data"])

    def colormap(self, map_type='height'):
        if map_type == 'height':
            return self.height_grid
        
        c = np.zeros((self.grid_size,self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if len(self.grid_3D[i][j]) == 0:
                    c[i][j] = 0
                    continue
                elif map_type == 'top':
                    c[i][j] = self.grid_3D[i][j][-1]
                elif map_type == 'avg':
                    c[i][j] = np.mean(self.grid_3D[i][j])
                elif map_type == 'mode':
                    c[i][j] = Counter(self.grid_3D[i][j]).most_common(1)[0][0]
        
        return c

    def plot_2D(self, color_type='height', cmap='hot'):
        """
        Plots the heights of the grain stacks by default, although 
        other color meanings are possible.
        """
        plt.imshow(self.colormap(color_type), cmap='hot')
        plt.colorbar()
        plt.show()

    def plot_3D(self, color_type='top', cmap='jet'):
        """
        color_type options:
            'top': the color indicates the type of the grain at the top of a stack
            'avg': the color indicates the average grain type in a stack
            'mode': the color indicates the most frequent grain type in a stack
            'height' the color indicates the height of a stack 
        """
        x = np.outer(range(self.grid_size), np.ones(self.grid_size))
        y = x.copy().T

        color_values = self.colormap(color_type) 
        m = plt.cm.ScalarMappable(norm=mcolors.Normalize(color_values.min(), color_values.max()), cmap=cmap)
        m.set_array([])
        fcolors = m.to_rgba(color_values)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(x, y, self.height_grid, facecolors=fcolors,
                       linewidth=0, antialiased=False, shade=False)
        mappable = plt.cm.ScalarMappable(cmap=cmap)
        mappable.set_array(color_values)
        plt.colorbar(mappable)
        plt.show()

    def plot_slice(self, slice_index=None, rot=False, cmap='jet'):
        """
        Plot a slice of the sandpile at "slice_index". If "rot" is True,
        the slice is taken in the z-axis instead of the x-axis.
        """

        # plot middle slice by default
        if slice_index == None:
            slice_index = self.grid_size // 2
            
        matrix = copy.deepcopy(self.grid_3D)
        tallest_stack = int(np.amax(self.height_grid))
        
        # pad with zeroes above stacks
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                matrix[i][j] = ([elem + 1 for elem in matrix[i][j]] + tallest_stack * [0])[:tallest_stack]

        if rot:
            matrix = np.rot90(matrix)

        slice_to_plot = np.array(matrix[slice_index])
        slice_to_plot = np.ma.masked_where(slice_to_plot == 0, slice_to_plot)
        
        plt.imshow(np.rot90(slice_to_plot), cmap=cmap)
        plt.colorbar()
        plt.show()
        
    def save(self, name = "Untitled"):
        with open(name + ".pkl", 'wb') as f:
            pickle.dump([self.data, self.height_grid, self.grid_3D, self.n_steps], f)
    
    def load(self, name = "Untitled"):
        with open(name + ".pkl", 'rb') as f:
            data, height_grid, grid_3D, n_steps = pickle.load(f)
            return data, height_grid, grid_3D, n_steps

if __name__ == "__main__":
    model = Sandpile_model(grid_size=20, n_steps=10000, crit_values=[4, 8], n_grain_types=2, boundary_con=False, init_method="random")
    model.run()

    # model.plot_time_series()
    # model.plot_size_probability(n_bins=100)
    
    # model.plot_slice()
    # model.plot_2D()
    # model.plot_3D(color_type='height')
    # model.plot_3D(color_type='top')
    # model.plot_3D(color_type='avg')
    # model.plot_3D(color_type='mode')
    # model.save("Sandpile_N_" + str(int(round(math.log10(model.n_steps), 0))) + "_CR_" + str(model.crit_values[0]) + str(model.crit_values[1])+ "_AM_" + model.init_method + "_GS_" + str(model.grid_size))
