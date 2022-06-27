# TODO

## 29 June
 - Load file, if file does not exist run model and save data
 - 

## 27 June

 - fitted parameters as a function of grain treshold for a singular grain
 - fitted parameter as a phase diagrom for two grain treshold

## 24 June

 - Grid initialization should be random. Prepare the system in an arbitrary stable configuration with z_i <= z^th for all i.
    Tot hoogte van laatste treshold en skip eerste x% aan stappen

 - Avalanche sizes of 2 do not seem to appear for a single grain
    Miss een artifact van het plaatsen in het centrum. Testen met random plaatsing. Dit is niet te zien bij meerdere grains omdat er dan maar 1 ding kan vallen

 - Grains are not falling at the boundaries when their height is larger than their treshold. This code should fix this:

    ```python
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
    ```
 - Powerlaw distribution with exponential falloff fit
    $$
    f(x) \propto x^{-\tau}a^x
    $$

### Results
 - Compare single grain with multi grains. Single grains of 2, 4, and 6 Multigrains of 2 and 6. 
 - Comparison between random, center, possibly corner and side. For multigrains of 2 and 6. 
 - Comparison between with and without boundaries. 

