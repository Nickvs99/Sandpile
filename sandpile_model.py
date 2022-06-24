import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import curve_fit

from plots import plot_time_series, plot_size_probability

class Sandpile_model:

    def __init__(self, grid_size=100, n_grain_types=2, crit_values=[1, 2], n_steps=10000, boundary_con=False, add_method="position", init_method="random", seed=None):
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
        if len(self.grid_3D[i][j]) == 0:
            return None
        
        # Get neighbour heights and whether they exist
        neighbour_heights, is_inside = self.get_neighbours(self.height_grid, [i, j])
        
        grain_type = self.grid_3D[i][j][-1]

        # Check whether an avalanche should occur
        if self.height_grid[i][j] - min(neighbour_heights) >= self.crit_values[grain_type] and self.height_grid[i][j] >= 4:

            top_grains = self.grid_3D[i][j][-4:]
            random.shuffle(top_grains)
            next_coords = [(i, j)]
            
            itter = 0
            for adj in range(-1, 2, 2):
                if is_inside[itter] == 1:
                    self.height_grid[i + adj][j] += 1
                    self.grid_3D[i + adj][j].append(top_grains[itter])
                    next_coords.append((i + adj, j))
                itter += 1

                if is_inside[itter] == 1:
                    self.height_grid[i][j + adj] += 1
                    self.grid_3D[i][j + adj].append(top_grains[itter])
                    next_coords.append((i, j + adj))
                itter += 1

            # implement the boundary condition and lower the pile at critical value
            if self.boundary_con:
                self.height_grid[i][j] -= sum(is_inside)
                for k in range(sum(is_inside)):
                    self.grid_3D[i][j].pop()
            else:
                self.height_grid[i][j] -= 4
                for k in range(4):
                    self.grid_3D[i][j].pop()

            return next_coords

        return None

    def update(self):
        
        new_grain_pos = self.get_position()   
        self.add_grain(new_grain_pos)

        # update the model simultaneously
        avalanche = 0
        control = 0

        avalanche_queue = [new_grain_pos]
        avalanche_size = 0

        while len(avalanche_queue) > 0:
            next_pos = avalanche_queue.pop()
            new_step = self.avalanche_step(next_pos[0], next_pos[1])
            if new_step != None:
                avalanche_size += 1

                avalanche_queue.extend(new_step)

        self.data[self.current_step] = avalanche_size

        self.current_step += 1

    def add_grain(self, position):

        grain_type = self.get_random_grain_type()
        self.grid_3D[position[0]][position[1]].append(grain_type)
        self.height_grid[position[0]][position[1]] += 1

        return position

    def get_random_grain_type(self):
        return np.random.randint(self.n_grain_types)

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
            
            randcords = [np.random.normal(cords[0], std), np.random.normal(cords[1], std)]

            return randcords

    def get_neighbours(self, matrix, coordinates):
        """
        find the neighbour values of a given cell
        Returns the values and gives the neighbours that exist given the dataset
        in order: left, upper, right and down neighbouring cells.
        """
        
        neighbours = np.zeros(4)
        real = np.zeros(4)

        if coordinates[0] >= 1:
            neighbours[0] = matrix[coordinates[0] - 1][coordinates[1]]
            real[0] += 1
        if coordinates[1] >= 1:
            neighbours[1] = matrix[coordinates[0]][coordinates[1] - 1]
            real[1] += 1

        if coordinates[0] < len(matrix) - 1:
            neighbours[2] = matrix[coordinates[0] + 1][coordinates[1]]
            real[2] += 1
        if coordinates[1] < len(matrix) - 1:
            neighbours[3] = matrix[coordinates[0]][coordinates[1] + 1]
            real[3] += 1
            
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

def fit_func(s, tau, a, C):
    """ A powerlaw distribution with exponential falloff. C is a normalization constant. """

    return C * s**(-tau) * a**s

def scaled_fit_func(x, tau, a, C):

    return np.log(fit_func(x, tau, a, C))

if __name__ == "__main__":

    model = Sandpile_model(grid_size=20, n_steps=100000, crit_values=[2, 4], n_grain_types=2, init_method="random")
    model.run()

    model.plot_time_series()
    model.plot_size_probability(n_bins=100)
