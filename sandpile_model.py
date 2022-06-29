import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from scipy.optimize import curve_fit
from scipy.stats import mode
from matplotlib import colors as mcolors
from collections import Counter
from itertools import chain

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
         - crit_values (list): The critical values of the grains. This list needs to be the same length
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
            np.random.shuffle(top_grains)

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
        # set values of cordinates if no values are given
        if cords[0] == -1:
            cords[0] = math.floor(self.grid_size/2)
        if cords[1] == -1:
            cords[1] = math.floor(self.grid_size/2)
            
        if self.add_method.lower() == "normal":
            
            # set value of standard deviation
            if std == -1:
                std = math.ceil(self.grid_size/12)
            
            randcords = [-1, -1]
            while min(randcords) < 0 or max(randcords) >= self.grid_size:
                randcords = [int(np.round(np.random.normal(cords[0], std))), int(np.round(np.random.normal(cords[1], std)))]
                
            return randcords
            
        if self.add_method.lower() == "random":
            randcords = [np.random.randint(self.grid_size - 1), np.random.randint(self.grid_size - 1)]
            return randcords
        
        if self.add_method.lower() == "position":
            return cords

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

        bin_centers = np.array([(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)])
        hist = np.array(hist)

        # Remove data entries where the data is zero, this removes the vertical line
        # And smoothens the data at very low probabilities due to missing avalanches
        bin_centers = bin_centers[hist != 0]
        hist = hist[hist != 0]

        return bin_centers, hist

#region plots
    def plot_time_series(self):

        ts, time_series_data = self.collect_time_series_data()

        plot_time_series([(ts, time_series_data)])

    def plot_size_probability(self, n_bins=20, show_fit=True):

        bin_centers, hist = self.collect_size_probability_data(n_bins=n_bins)
        
        if show_fit:
            popt, pcov = self.calc_fit_parameters(bin_centers, hist)   
            plt.plot(bin_centers, fit_func(bin_centers, *popt), label=f"fit: tau={popt[0]:.3f}, a={popt[1]:.3f}")

            labels = ["raw data"]
        else:
            labels = []

        plot_size_probability([(bin_centers, hist)], labels=labels)

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
                    c[i][j] = self.grid_3D[i][j][-1]+1
                elif map_type == 'avg':
                    c[i][j] = np.mean([self.crit_values[item] for item in self.grid_3D[i][j]])
                elif map_type == 'mode':
                    c[i][j] = Counter(self.grid_3D[i][j]).most_common(1)[0][0]+1
        
        return c

    def plot_2D(self, frame=None, animate=False, show=True, color_type='height', cmap='hot'):
        """
        Plots the heights of the grain stacks by default, although 
        other color meanings are possible.
        """
        if animate:
            self.update()
            self.im.set_array(self.colormap(color_type))
            if self.current_step % 100 == 0:
                print(f"Step {self.current_step} of {self.n_steps}", end="\r")
            return self.im,
        else:
            self.im = plt.imshow(self.colormap(color_type), cmap=cmap, animated=True)
            plt.colorbar()
            if color_type == 'height':
                plt.title("Height of stacks")
            elif color_type == 'top':
                plt.title("Top grains")
            elif color_type == 'avg':
                plt.title("Average threshold in stack")
            elif color_type == 'mode':
                plt.title("Most frequent grain type in stack")
                
            if show:
                plt.show()

            

    def animate_2D(self, ani_func, fargs, color_type='top', cmap='gnuplot', save=False):
        fig = plt.figure()
        ani_func(*(None, False,)+fargs)
        self.ani = animation.FuncAnimation(fig, ani_func, fargs=(True,)+fargs, frames=self.n_steps-2, interval=1, blit=True, repeat=False)
        
        if save != False:
            import matplotlib as mpl 
            mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\sande\\Desktop\\ffmpeg\\bin\\ffmpeg.exe'
            f = f"6_3.mp4" 
            writervideo = animation.FFMpegWriter(fps=60)
            # self.ani.save('2osc.mp4', writer="ffmpeg")
            self.ani.save(f, writer=writervideo)
        else:
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

    def plot_slice(self, frame=None, animate=False, show=True, cmap='jet', slice_index=None, rot=False):
        """
        Plot a slice of the sandpile at "slice_index". If "rot" is True,
        the slice is taken in the z-axis instead of the x-axis.
        """
        # plot middle slice by default
        if slice_index == None:
            slice_index = self.grid_size // 2
            
        matrix = copy.deepcopy(self.grid_3D)
        
        # pad with zeroes above stacks
        if animate or not show:
            padding = self.grid_size // 3 * max(self.crit_values)
        else:
            padding = self.tallest_stack()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                matrix[i][j] = ([elem + 1 for elem in matrix[i][j]] + padding * [0])[:padding]

        if rot:
            matrix = np.rot90(matrix)

        slice_to_plot = np.array(matrix[slice_index])
        slice_to_plot = np.ma.masked_where(slice_to_plot == 0, slice_to_plot)
        
        if animate:
            self.im.set_array(np.flipud(np.rot90(slice_to_plot)))
            if self.current_step % 100 == 0:
                print(f"Step {self.current_step} of {self.n_steps}", end="\r")
            for _ in range(1):
                self.update()
            return self.im,
        else:
            self.im = plt.imshow(np.flipud(np.rot90(slice_to_plot)), cmap=cmap, animated=True, origin="lower")
            if self.n_grain_types > 1:
                plt.colorbar()
            if show:
                plt.show()

    def tallest_stack(self):
        return int(np.amax(self.height_grid))

    def voxel_plot(self):
        axes = [self.grid_size, self.grid_size, self.tallest_stack()]
        grain_grid = np.ones(axes, dtype=np.bool)
        color_grid = np.zeros(axes + [4], dtype=np.float32)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.tallest_stack()):
                    try:
                        if self.grid_3D[i-1][j][k]+1 and self.grid_3D[i+1][j][k]+1 and self.grid_3D[i][j-1][k]+1 and self.grid_3D[i][j+1][k]+1 and self.grid_3D[i][j][k+1]+1 and 1/k:
                            grain_grid[i][j][k] = 0
                    except:
                        try:
                            color_grid[i][j][k] = ([np.clip(self.grid_3D[i][j][k] % 2 + 1, 0, 1), np.clip(self.grid_3D[i][j][k] % 2, 0, 1), np.clip(self.grid_3D[i][j][k] % 3 - 1, 0, 1), 1])
                        except:
                            grain_grid[i][j][k] = 0

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(grain_grid, facecolors=color_grid)
        ax.set_box_aspect(np.ptp(np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz']), axis = 1))
        plt.show()
        
    def proportions(self, array):
        flattened=array
        while type(flattened[0]) == list:
            flattened = list(chain.from_iterable(flattened))
        unique, counts = np.unique(flattened, return_counts=True)
        counts = dict(zip(unique, counts))
        for t in range(self.n_grain_types):
            if t not in counts:
                counts[t] = 0
        counts = np.array([item[1] for item in sorted(counts.items())])
        return counts / sum(counts)
    
    def proportions_over_time(self, prop_type=("top")):
        props_top = []
        props_total = []

        while self.current_step < self.n_steps:
            self.update()
            if "top" in prop_type:
                props_top.append(self.proportions(self.colormap('top')))
            if "total" in prop_type:
                props_total.append(self.proportions(self.grid_3D))

            if self.current_step % 1000 == 0:
                print(f"Step {self.current_step} of {self.n_steps}", end="\r")

        if "top" in prop_type:
            plt.plot(range(len(props_top)), props_top, label=[f"Threshold: {self.crit_values[i-1]}" if i > 0 else "Empty cells" for i in range(len(props_top[0]))])
            plt.legend()
            plt.ylim(bottom=0)
            plt.xlabel("Steps")
            plt.ylabel("Proportion")
            plt.title("Top grains")
            plt.show()

        if "total" in prop_type:
            plt.plot(range(len(props_total)), props_total, label=[f"Threshold: {self.crit_values[i]}" for i in range(len(props_total[0]))])
            plt.legend()
            plt.title("All grains")
            plt.ylim(bottom=0)
            plt.xlabel("Steps")
            plt.ylabel("Proportion")
            plt.show()
#endregion

#region save load
    def save(self):

        data_dir = "data"
        
        # Check if data folder exists
        if not os.path.isdir(data_dir):
            print("Creating data directory...")

            os.mkdir(data_dir)

        # Check if model has completed
        if self.current_step != self.n_steps:
            print("WARNING: You are saving a model which has not completed.")

        file_name = self.get_file_name()

        save_location = os.path.join(data_dir, file_name)

        print(f"Saving data to {save_location}...")
        with open(save_location, 'wb') as f:
            pickle.dump(self, f)
    
    def load_or_run(self):
        """
        Attempts to load an already saved instance with the current parameters. 
        If no load file exists with these parameters, the instance will run 
        and save itself. 
        """

        data_dir = "data"
        file_name = self.get_file_name()
        load_location = os.path.join(data_dir, file_name)

        if os.path.isfile(load_location):
            
            print(f"Loading data {self}...")
            with open(load_location, 'rb') as f:
                obj = pickle.load(f)

                # Copies the values from the saved instance into the current instance
                self.__dict__ = copy.deepcopy(obj.__dict__)


            # Check if model has completed
            if self.current_step != self.n_steps:
                print("WARNING: You are loading a model which has not completed.")

        else:
            print(f"Running model {self}...")
            self.run()
            self.save()
    
    def get_file_name(self):

        return f"data_N_{self.n_steps}_GS_{self.grid_size}_n_{self.n_grain_types}_GO_{self.grain_odds}_CR_{self.crit_values}_ADD_{self.add_method}_INIT_{self.init_method}_B_{self.boundary_con}.pickle"
    
    def __str__(self):

        # Return file name without the extension
        return self.get_file_name()[:-7]
#endregion

if __name__ == "__main__":
    
    # grain_tresholds = np.arange(1, 11)

    # for i, grain_treshold1 in enumerate(grain_tresholds):

    #     # Since the results are symmetric not all possibilies need to be tested
    #     # eg. the tresholds (1, 2) and (2, 1) should give the same results
    #     for j, grain_treshold2 in enumerate(grain_tresholds[:i + 1]):
        
    #         model = Sandpile_model(grid_size=32, n_steps=1000000, crit_values=[grain_treshold1, grain_treshold2], n_grain_types=2, init_method="random", add_method="random")
    #         model.load_or_run()

    model = Sandpile_model(grid_size=32, n_steps=10000, crit_values=[6,3], n_grain_types=2, add_method="position", boundary_con=False, init_method="random")
    # model.load_or_run()

    # model.plot_time_series()
    # model.plot_size_probability(n_bins=100)

    # todo: voxels, filmpje avalanches, grafiekje proportion over tijd, kleur threshold ipv type, corner 3D
    
    # model.animate_2D(model.plot_2D, (False, "top", "gnuplot"), save=True)
    model.animate_2D(model.plot_slice, (False, 'brg'), save=True)
    # model.proportions_over_time(("top", "total"))

    model.voxel_plot()
    model.plot_slice(cmap='gnuplot')
    model.plot_3D(color_type='top', cmap='gnuplot')
    model.plot_2D(color_type='height')
    model.plot_2D(color_type='top', cmap='gnuplot')
    model.plot_2D(color_type='avg')
    model.plot_2D(color_type='mode', cmap='gnuplot')

    
    # model.save("Sandpile_N_" + str(int(round(math.log10(model.n_steps), 0))) + "_CR_" + str(model.crit_values[0]) + str(model.crit_values[1])+ "_AM_" + model.init_method + "_GS_" + str(model.grid_size))
