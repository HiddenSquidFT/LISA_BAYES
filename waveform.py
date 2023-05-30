import BBHX_PhenomD
import numpy as np
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.filter import resample_to_delta_t, highpass, matched_filter, sigma
import pycbc.psd
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
import pycbc.waveform.spa_tmplt as pyspt
from pycbc import frame, types, fft, waveform
from pycbc.types import TimeSeries, FrequencySeries
from bbhx.waveformbuild import BBHWaveformFD
from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
from bbhx.response.fastfdresponse import LISATDIResponse
from bbhx.utils.constants import *
from bbhx.utils.transform import *
import BBHX_PhenomD
import pycbc.noise
from bbhx.utils.transform import LISA_to_SSB, SSB_to_LISA
from pycbc.conversions import  mchirp_from_mass1_mass2
import astropy.coordinates
import pickle




def waveform_gen(number_of_mergers, mass_low, mass_high, write_to_file=False):
    ### ARGUEMENTS ####
    ### number_of_mergers: The amount of waveforms you want to generate between mass_low and mass_high. ###
    ### mass_low, mass_high: The lower and upper limits of the component mass of the merger you want to generate, the code will ensure m1>m2. ###
    ### write_to_file: If True it will write the gwf files, set to False by default because the files are large and will take time to write. ###
    
    
    from circle import create_dist
    
    
    circle = create_dist(7,0.5,500,0,0) ### Generate circle from circle.py ###
    
    lam_f,beta_f,z_f = zip(*circle[0]) ### Unzip our false positions, we zipped to surpress the output of create_dist when calling waveform_gen ###
    
    
    
    param_names =["m1","m2","chi1z","chi2z","dist","inc","beta","lam","psi"]  ### Used for dictionaries ##
    wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=True))
    
    zeros = TimeSeries(np.zeros(int(31536000)),delta_t=5, epoch=0)   ### Initialise frequencies and waves ###
    
    f = np.arange(len(zeros) / 2 + 1)
    
    f = f * 1/(31536000) # Multiplied by 1/duration (in seconds)
    
    dict_of_mergers_A = dict()   ### This will become a dictionary of dictionaries, with information on all the mergers ##
    
    false_dict = dict()


    print(f) # Should have last value equal to delta_t / 2
    freq_new = f # Remove first 0 point from here
    m_arr = np.logspace(np.log10(mass_low),np.log10(mass_high),number_of_mergers) ### Initialse mass array ###



    f_ref = 0.0  # let phenom codes set f_ref -> fmax = max(f^2A(f))
    phi_ref = 0.0 # phase at f_ref
    for i,m in enumerate(m_arr):
        
        
        merger_values={}
    ## WE ARE GOING TO GENERATE TWO WAVEFORMS FOR MODEL COMPARISON, THAT ARE SLIGHTLY APART TRANSVERSALLY BUT VARY IN REDSHIFT ##


        ### SAME PARAMETERS FOR BOTH POTENTIAL LOCATIONS ###
        chi1z = np.random.uniform(-0.99,0.99)
        chi2z = np.random.uniform(-0.99,0.99)
        m1 = np.random.uniform(1,5)*m    ### Ensure m1>m2, this also sets 1<q<5
        m2 = m
        dist = cosmo.luminosity_distance(0.5).value  * PC_SI * 1e6  ## ANCHOR ONE MERGER TO A DISTANCE ##
        inc = circle[2]  ### The inclination is saved in the output of cirlce.py ###
        psi = np.pi/5 # polarization angle 
        #################################


         
        t_obs_start = 1
        
        #### FIX SKY POSITION for all model A (true) mergers #####
        
        beta = inc   # Ecliptic latitude     
        lam = circle[1]  # Ecliptic longitude, also stored from circle.py ##
        t_ref = 4800021
        coord = SSB_to_LISA(t_ref,lam,beta,psi)   ### TRANSLATE TO LISA COORDS FOR INFERENCE ###
        #print(coord)
        wave = wave_gen(m1, m2, chi1z, chi2z,
                                  dist, phi_ref, f_ref, inc, lam,   
                                  beta, psi, t_ref, freqs=f,   
                                  direct=False, fill=True, squeeze=True, length=1024)[0]   ## This part is the generation code ##


        param_values =[m1,m2,chi1z,chi2z,dist,inc,coord[2],coord[1],coord[3]]   
        for jdx,j in enumerate(param_names):
            merger_values[j] = param_values[jdx]  ### WRITE NESTED DICTIONARY INTO I'TH ELEMENT OF DICTIONARY ###
        dict_of_mergers_A[i] = merger_values
        

        ### Not the most efficient method but it writes the config files needed for inference on the command line ###
        write = open("config_"+str(i)+str("_GAL_A")+".ini","w")   
        write.write('''
        [data]
        instruments = LISA_A LISA_E LISA_T
        trigger-time = 4800021
        analysis-start-time = -4800021
        analysis-end-time = 26735978
        pad-data = 0
        sample-rate = 0.2
        psd-file= LISA_A:A_psd.txt LISA_E:E_psd.txt LISA_T:T_psd.txt
        frame-files = LISA_A:Response_A_{i}_A.gwf LISA_E:Response_E_{i}_A.gwf LISA_T:Response_T_{i}_A.gwf
        channel-name = LISA_A:LA:LA LISA_E:LE:LE LISA_T:LT:LT


        [model]
        name = relative
        low-frequency-cutoff = 0.0001
        high-frequency-cutoff = 1e-2
        epsilon = 0.01
        mchirp_ref = {mc}
        q_ref = {q}
        mass1_ref = {m1}
        mass2_ref = {m2}
        tc_ref = {tc}               
        distance_ref = {dist}
        spin1z_ref = {spin1}
        spin2z_ref = {spin2}
        inclination_ref = {inc}
        eclipticlongitude_ref = {long}
        eclipticlatitude_ref = {lat}


        [variable_params]
        distance =  
        mchirp = 
        spin1z = 
        spin2z = 
        inclination = 


        [static_params]
        approximant = BBHX_PhenomD
        coa_phase = 0
        polarization = {pol}
        t_obs_start = 31536000
        q = {q}
        tc = {tc}
        eclipticlongitude = {long}
        eclipticlatitude = {lat}


        [prior-mchirp]
        name = uniform
        min-mchirp = {mc_min}
        max-mchirp = {mc_max}


        [prior-distance]
        name = uniform
        min-distance = {dist_min}
        max-distance = {dist_max}


        [prior-spin1z]
        name = uniform
        min-spin1z = -0.99
        max-spin1z = 0.99

        [prior-spin2z]
        name = uniform
        min-spin2z = -0.99
        max-spin2z = 0.99

        [prior-inclination]
        name = sin_angle

    ;    [prior-eclipticlongitude]
    ;    name = uniform
    ;    min-eclipticlongitude = 0.0
    ;    max-eclipticlongitude = 6.28318530718

    ;    [prior-eclipticlatitude]
    ;    name = uniform
    ;    min-eclipticlatitude = -1.5707963267948966
    ;    max-eclipticlatitude = 1.5707963267948966

    ;    [prior-tc]
    ;    name = uniform
    ;    min-tc = {tc_min}
    ;    max-tc = {tc_max}


        [waveform_transforms-mass1+mass2]
        name = mchirp_q_to_mass1_mass2

        [sampler]
        name = dynesty
        dlogz = 0.1
        nlive = 750
        pool = pool
        queue_size= 10


        ; NOTE: While this example doesn't sample in polarization, if doing this we
        ; recommend the following transformation, and then sampling in this coordinate
        ;
        ; [waveform_transforms-polarization]
        ; name = custom
        ; inputs = better_pol, eclipticlongitude
        ; polarization = better_pol + eclipticlongitude





    ''' .format(mc = mchirp_from_mass1_mass2(m1,m2), m1=m1, m2=m2, inc = inc,


                mc_min = 0.6* mchirp_from_mass1_mass2(m1,m2),
                mc_max = 1.5*mchirp_from_mass1_mass2(m1,m2),
                q=m1/m2,

                i=i, tc=coord[0], tc_min = 0.8*coord[0] , tc_max = 1.2*coord[0],


                dist = dist / (PC_SI*1e6), 

                spin1 = chi1z, 
                spin2 = chi2z, 

                long = coord[1], 
                lat = coord[2],
                pol = coord[3],


               dist_min = 0.5*(dist / (PC_SI*1e6)), dist_max = 1.5*(dist / (PC_SI*1e6))))
        write.close() 

        beta_false = np.random.choice(beta_f)

        lam_false = np.random.choice(lam_f)### NOW WRITE FALSE PARAMETERS INTO ANOTHER CONFIG FILE ###

        dist_false = np.random.choice(z_f)
        #cosmo.luminosity_distance(np.random.choice(z_rand)).value  * PC_SI * 1e6

       ## This sets up our model B (false) for model comparison in the next step ##
       ## PRIOR VOLUMES MUST BE IDENTICAL ##
        coord_B = SSB_to_LISA(t_ref,lam_false,beta_false,psi) ## Translate false coordinates to LISA frame ##

        print(dist,dist_false)

        merger_values_false={}
        param_values_false =[m1,m2,chi1z,chi2z,dist_false,inc,coord_B[2],coord_B[1],coord[3]]



        for jdx,j in enumerate(param_names):
            merger_values_false[j] = param_values_false[jdx]  ### WRITE NESTED DICTIONARY INTO i'th ELEMENT OF DICTIONARY ###
        false_dict[i] = merger_values_false
        #print(false_dict)               



       ### Now we write the config containing the incorrect parameters ###
        write = open("config_"+str(i)+str("_GAL_B")+".ini","w")
        write.write('''
        [data]
        instruments = LISA_A LISA_E LISA_T
        trigger-time = 4800021
        analysis-start-time = -4800021
        analysis-end-time = 26735978
        pad-data = 0
        sample-rate = 0.2
        psd-file= LISA_A:A_psd.txt LISA_E:E_psd.txt LISA_T:T_psd.txt
        frame-files = LISA_A:Response_A_{i}_A.gwf LISA_E:Response_E_{i}_A.gwf LISA_T:Response_T_{i}_A.gwf
        channel-name = LISA_A:LA:LA LISA_E:LE:LE LISA_T:LT:LT

        [model]
        name = relative
        low-frequency-cutoff = 0.0001
        high-frequency-cutoff = 1e-2
        epsilon = 0.01
        mchirp_ref = {mc}
        q_ref = {q}
        mass1_ref = {m1}
        mass2_ref = {m2}
        tc_ref = {tc}               
        distance_ref = {dist}
        spin1z_ref = {spin1}
        spin2z_ref = {spin2}
        inclination_ref = {inc}
        eclipticlongitude_ref = {long}
        eclipticlatitude_ref = {lat}



        [variable_params]
        distance =  
        mchirp = 
        spin1z = 
        spin2z = 
        inclination = 


        [static_params]
        approximant = BBHX_PhenomD
        coa_phase = 0
        polarization = {pol}
        t_obs_start = 31536000
        q = {q}
        tc = {tc}
        eclipticlongitude = {long}
        eclipticlatitude = {lat}


        [prior-mchirp]
        name = uniform
        min-mchirp = {mc_min}
        max-mchirp = {mc_max}


        [prior-distance]
        name = uniform
        min-distance = {dist_min}
        max-distance = {dist_max}


        [prior-spin1z]
        name = uniform
        min-spin1z = -0.99
        max-spin1z = 0.99

        [prior-spin2z]
        name = uniform
        min-spin2z = -0.99
        max-spin2z = 0.99

        [prior-inclination]
        name = sin_angle

    ;    [prior-eclipticlongitude]
    ;    name = uniform
    ;    min-eclipticlongitude = 0.0
    ;    max-eclipticlongitude = 6.28318530718

    ;    [prior-eclipticlatitude]
    ;    name = uniform
    ;    min-eclipticlatitude = -1.5707963267948966
    ;    max-eclipticlatitude = 1.5707963267948966



    ;    [prior-tc]
    ;    name = uniform
    ;    min-tc = {tc_min}
    ;    max-tc = {tc_max}


        [waveform_transforms-mass1+mass2]
        name = mchirp_q_to_mass1_mass2

        [sampler]
        name = dynesty
        dlogz = 0.1
        nlive = 750
        pool = pool
        queue_size= 10


        ; NOTE: While this example doesn't sample in polarization, if doing this we
        ; recommend the following transformation, and then sampling in this coordinate
        ;
        ; [waveform_transforms-polarization]
        ; name = custom
        ; inputs = better_pol, eclipticlongitude
        ; polarization = better_pol + eclipticlongitude





    ''' .format(mc = mchirp_from_mass1_mass2(m1,m2), m1=m1, m2=m2, inc = inc,


                mc_min = 0.6* mchirp_from_mass1_mass2(m1,m2),
                mc_max = 1.5*mchirp_from_mass1_mass2(m1,m2),
                q=m1/m2,

                i=i, tc=coord[0], tc_min = 0.8*coord[0], tc_max = 1.2*coord[0],

                dist = dist/ (PC_SI*1e6), 

                spin1 = chi1z, 
                spin2 = chi2z, 

                long = coord_B[1], 
                lat = coord_B[2],
                pol = coord[3],


               dist_min = 0.5*(dist / (PC_SI*1e6)), dist_max = 1.5*(dist / (PC_SI*1e6))))
        write.close() 

        print(lam,lam_false)
        print(beta,beta_false)
        print(("angular seperation is",astropy.coordinates.angular_separation(coord[1],coord[2],coord_B[1],coord_B[2])))

       ## This is where we translate our waveforms into the class types accepted by PyCBC, so we can write the .gwf files ###
        wave_A_response_GAL_A = FrequencySeries(wave[0], delta_f=1/(31536000), epoch=0).to_timeseries(delta_t=1)
        wave_E_response_GAL_A = FrequencySeries(wave[1], delta_f=1/(31536000), epoch=0).to_timeseries(delta_t=1)
        wave_T_response_GAL_A = FrequencySeries(wave[2], delta_f=1/(31536000), epoch=0).to_timeseries(delta_t=1)

        if write_to_file == True:
            frame.write_frame("./Response_A_{}_A.gwf".format(i),"LA:LA",wave_A_response_GAL_A)
            frame.write_frame("./Response_E_{}_A.gwf".format(i),"LE:LE",wave_E_response_GAL_A)
            frame.write_frame("./Response_T_{}_A.gwf".format(i),"LT:LT",wave_T_response_GAL_A)
        ### Pickle our dictionaries ##
        with open('merge_dict_A.pkl', 'wb') as f:
            pickle.dump(dict_of_mergers_A, f)  
    
        with open('false_dict.pkl', 'wb') as f:
            pickle.dump(false_dict, f)
        
        
        write = open("bash.sh","w")  ## WRITE THE BASH FILE THAT PERFOMS THE INFRENCE ##
        write.write('''
            #!/bin/sh


    for i in {0..19}         
    do
        echo "Starting_A_$i"
          pycbc_inference \
        --config-files config_"$i"_GAL_A.ini \
        --output-file lisa_smbhb_GAL_A_$i.hdf \
        --force \
        --nprocesses 10 \
        --verbose
        echo "Done $file"





        echo "Starting_B_$i"
        pycbc_inference \
        --config-files config_"$i"_GAL_B.ini \
        --output-file lisa_smbhb_GAL_B_$i.hdf \
        --force \
        --nprocesses 10 \
        --verbose
        echo "Done $file"
        done
        
        
        
        
        ''')
    write.close()
    
    # :) # 
