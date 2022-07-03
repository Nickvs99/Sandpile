# Sandpile

By: Ruben Bonneur, Sander Broos and Nick van Santen

This project extends the basic sandpile model by introducing multiple grain types. This extended sandpile model is then studied for the rise of self-organized criticality, an altered avalanche distribution, and possible emergent behaviour.  

## Project structure

The import files and folders:
 - data - A folder containing already computed model configurations. These can then be loaded in to remove the need to recalculate an instance of the model.
 - figures - The final figures used in the presentation, which can be viewed in ```Presentation.pdf```.
 - ```data_collection.py``` - A python script that utilizes multi-processing to reduce the computational time of a model.
 - ```plots.py``` - A python script that handles the plotting of the time series and avalanche distribution.
 - ```results.ipynb``` - A jupyter notebook with the functionality to compute and display the obtained figures. 
 - ```sandpile_model.py``` - The extended version of the basic sandpile model. It also includes several methods to plot the 3D structure of a sandpile, as well as the saving and loading functionality. 
