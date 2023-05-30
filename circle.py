import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo




def create_dist(diameter, redshift,n,
                lam,beta,           
                plot = False, save = False):  ### Generate a random uniform spherical distribution ###
   
    ### ARGUEMENTS: ###
    #####################################################################################################################################
    ### diameter: The diameter of the distribution, best to define in MPC as the context for the distribution is galaxy clusters. #######
    ### redshift: The distance to the object, usually between 0 and 20, the code will translate this to MPC.                      #######
    ### n: The number of samples you want generated on the sphere.                                                                #######
    ### lam,beta: The ecliptic longitude and latitude respectively: LIMITS: 0<lam<pi/2 ; -pi/2 < beta < pi/2.                     #######
    ### plot: If true, it will plot the distribution, useful for checking everything worked.                                      #######
    ### save: If true, it will save the distribution as an numpy array in PWD.                                                    #######
    #####################################################################################################################################
    if lam<0 or lam>np.pi * 2:
      raise Exception("Lambda must be between 0 and 2pi")
      
    if beta<-np.pi/2 or beta>np.pi/2:
      raise Exception("Beta must be between -pi/2 and pi/2")
    radius = diameter / cosmo.luminosity_distance(redshift).value # rads # 
    x = np.random.uniform(low=-2*radius, high=2*radius, size=n*50) ## Hopefully this guarantees you will always have enough points within the sphere ##
    y = np.random.uniform(low=-2*radius, high=2*radius, size=n*50)
    z = np.random.uniform(low=-2*radius, high=2*radius, size=n*50)

    ## Rejection sampling to exclude points not on the sphere ##

    r = np.sqrt(x**2 + y**2)
    x = x[r <= 1.2*radius]
    y = y[r <= 1.2*radius]
    z = z[r <= 1.2*radius]

     ## Scaling ##
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
    

