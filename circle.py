import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo




def create_dist(diameter, redshift,n,
                lam,beta,           
                plot = False, save = False):  ### Generate a random uniform spherical distribution ###
    
    
    radius = diameter / cosmo.luminosity_distance(redshift).value # rads # 
    x = np.random.uniform(low=-2*radius, high=2*radius, size=20000)
    y = np.random.uniform(low=-2*radius, high=2*radius, size=20000)
    z = np.random.uniform(low=-2*radius, high=2*radius, size=20000)

    # Reject points that are outside the sphere #

    r = np.sqrt(x**2 + y**2)
    x = x[r <= 1.2*radius]
    y = y[r <= 1.2*radius]
    z = z[r <= 1.2*radius]

        # Scale points to lie on the surface of the sphere
    r = np.sqrt(x**2 + y**2 + z**2)
    
    x = radius * x / r + lam
    
    if beta == 0:
        y = radius * y / r + beta + 0.035
    else:
        y = radius * y / r + beta
    z = radius * z / r + cosmo.luminosity_distance(redshift).value
    
    true_x = lam
    true_y = beta
    
    
    x_rand = x[:n]
    y_rand = y[:n]
    z_rand = z[:n]
    
    
    if plot==True:
        plt.scatter(x_rand,y_rand)
        
        
    if save==True:
        np.savez("circle_dist.npz",x = x_rand, y = y_rand,z = z_rand )
    zip_coord = zip(x_rand,y_rand,z_rand)
    return list(zip_coord),true_x,true_y
    

