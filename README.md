# LISA_BAYES
Perform Bayesian analysis on clusters of galaxies
There are a number of pre-reqs to get started:
First follow the instructions here:
https://github.com/mikekatz04/BBHx
This will install all the packages needed like Astropy, SciPy etc.

You will also need the pandas package:
https://pandas.pydata.org/docs/getting_started/install.html

Then install the PyCBC plugin within the conda enviroment you just made:
https://github.com/ConWea/BBHX-waveform-model
Or in command line: 
pip install git+https://github.com/ConWea/BBHX-waveform-model.git

Run waveform.py first, it will call on circle.py to create a data set and then generate the waveforms. It will also create a bash file that contains the instructions for performing the inference.
RESULTS_MASTER is where you can see the results visualised.




CIRLCE:

Circle.py generates a circluar distribution of galaxies within a cluster where $d_\theta = D/r$ where r is the distance to the distribution and D is the physical size of the distribution. You can use it by itself but it is used in waveform.py.

WAVEFORM:

Waveform.py generates the waveforms (and can save the .gwf file) based on the parameters from cirlce.py, you define the mass range.
