import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import random

class Sandpile_model:

    def __init__(self, grid_size=100, n_grain_types=2, crit_values=[1, 2], n_steps=10000, boundary_con=False, add_method="position", seed=None):
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

        self.initialize_grids()

        self.current_step = 0
        self.data = np.zeros(self.n_steps)

    def initialize_grids(self):
        """
        Initalizes both the height_grid and grid_3D.
        height_grid stores the height at each location i,j
        grid_3D stores the grains at each location i,j
        """

        self.height_grid = np.zeros((self.grid_size, self.grid_size))
        
        self.grid_3D = []
        for i in range(self.grid_size):
            self.grid_3D.append([])
            
            for j in range(self.grid_size):
                self.grid_3D[i].append([])

    def run(self):

        while self.current_step < self.n_steps:
            self.update()

            if self.current_step % 1000 == 0:
                print(self.current_step)

    def update(self):
        
        self.add_grain()

        # update the model simultaneously
        avalanche = 0
        control = 0
        
        while True:
            
            height_grid_copy = copy.deepcopy(self.height_grid)
    
            for i in range(0, len(self.height_grid)):
                for j in range(0, len(self.height_grid[0])):
                    
                    # Check if a grain is present at location i,j
                    if len(self.grid_3D[i][j]) == 0:
                        continue
                    
                    # Get neighbour heights and wheter they exist
                    neighbour_heights, real = self.get_neighbours(height_grid_copy, [i, j])
                    
                    grain_type = self.grid_3D[i][j][-1]

                    # Check wheter an avalanche should occur
                    if self.height_grid[i][j] - min(neighbour_heights) >= self.crit_values[grain_type] and height_grid_copy[i][j] >= 4:
                        
                        avalanche += 1

                        # add to surrounding area
                        order = [-1, -2, -3, -4]
                        random.shuffle(order)
                        
                        itter = 0
                        for adj in range(-1, 2, 2):
                            if real[itter] == 1:
                                self.height_grid[i+ adj][j] += 1
                                self.grid_3D[i+ adj][j].append(self.grid_3D[i][j][order[itter]])
                            itter += 1

                            if real[itter] == 1:
                                self.height_grid[i][j + adj] += 1
                                self.grid_3D[i][j + adj].append(self.grid_3D[i][j][order[itter]])
                            itter += 1

                        # implement the boundary condition and lower the pile at critical value
                        if self.boundary_con:
                            self.height_grid[i][j] -= sum(real)
                            for k in range(sum(real)):
                                self.grid_3D[i][j].pop()
                        else:
                            self.height_grid[i][j] -= 4
                            for k in range(4):
                                self.grid_3D[i][j].pop()
            
            # Check if the avalanche has stopped propagating
            if control == avalanche:
                self.data[self.current_step] = avalanche
                break
                
            control = avalanche

        self.current_step += 1

    def add_grain(self):

        position = self.get_position()
        grain_type = self.get_random_grain_type()
        self.grid_3D[position[0]][position[1]].append(grain_type)
        self.height_grid[position[0]][position[1]] += 1

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
        
        neighbours = list()
        real = np.zeros(4)
        
        if coordinates[0] >= 1:
            neighbours.append(matrix[coordinates[0] - 1][coordinates[1]])
            real[0] += 1
        if coordinates[1] >= 1:
            neighbours.append(matrix[coordinates[0]][coordinates[1] - 1])
            real[1] += 1
        
        if coordinates[0] < len(matrix) - 1:
            neighbours.append(matrix[coordinates[0] + 1][coordinates[1]])
            real[2] += 1
        if coordinates[1] < len(matrix) - 1:
            neighbours.append(matrix[coordinates[0]][coordinates[1] + 1])
            real[3] += 1
            
        return neighbours, real

    def plot_time_series(self):

        avalanche_sizes = np.array(self.data)

        max_avalanche_size = np.amax(avalanche_sizes)

        # Normalize the avalanche sizes
        avalanche_sizes = avalanche_sizes / max_avalanche_size

        ts = np.arange(self.n_steps)
        plt.vlines(ts, 0, avalanche_sizes)

        plt.xlabel("t")
        plt.ylabel(r"$s/s_{max}$")

        plt.xlim(0, self.n_steps)
        plt.ylim(0,1)

        plt.show()


if __name__ == "__main__":

    model = Sandpile_model(grid_size=10, n_steps=1000, crit_values=[1, 3])
    model.run()

    print(model.grid_3D)
    print(model.height_grid)

    model.plot_time_series()
