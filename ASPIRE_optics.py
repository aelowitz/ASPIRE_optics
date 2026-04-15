

'''
This optics code was written by Amy Lowitz (lowitz@arizona.edu).  
The key underlying Gaussian optics equations are derived in many textbooks and 
papers and are standard to the field.  In most cases, however, I point to specific 
equation numbers from Paul Goldsmith's textbook, Quasioptical Systems, in inline comments, 
as a guidepost to any user needing an entry-point for learning more about the relevant
physical underpinnings.  


Author: Amy Lowitz - Spring 2025

Updates:
    Late June 2025 (AEL):   - Complete refactor to make agnostic to number of mirrors.
                            - Significantly reduces code redudndancies between forward and reverse functionality
                            - Got rid of old hyper-specific "optimizing" and constraint solving functions



Summary:
This module calculates and plots various quantities useful in the design of a Gaussian optical
system.  This code was developed pecifically with the ASPIRE receiver (Advanced 
South Pole Integrated Receiver for EHT) in mind, but could be adapted to more general 
systems.  


Usage:
    1) edit hardcoded mirror parameters in generate_parameter_dict()
    2) run main()





#TODO: - Use a config file to set mirror params instead of hard-coding in generate_parameter_dict()
       - General cleanup, block comments



'''



import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opt
import copy
import warnings as wa

def generate_parameter_dict():
    '''
        ##ALMA band 6: 211-275 GHz.  Band 7: 275-373.  Band 2: 67-116.
    
        Takes the hard-coded parameters at the top of this function and puts them into a dictionary 
        to be referenced by the beam propagation code. 
        
        Edit between the "EDIT BELOW HERE" comment and the "EDIT ABOVE HERE" comment.
    
    '''
    
    #physics parameters and unchangable telescope and horn parameters
    c = 299792458 #m/s, speed of light in a vacuum
    f1 = 8003.21 #mm, focal length of the primary, from Padin 2008
    f2 = 818.2808 #mm, effective (thin lens equiv) focal length of the secondary, from Stark memo 2018.  f1 = 1310.0 mm, f2 = 2180.0 mm, 
    d_12 = 8003.21 + 1300 # mm, distance between the primary and secondary, from Stark memo. Focal length + distance from prime focus to M2
    a_h_B6 = 3.54 #mm, horn aperture radius, from Greer memo and confirmed by personal communication with John Effland
    a_h_B7 = 3.00225 # horn aperture radius, from ALMA Band 7 Cartridge Preliminary design report p.29
    R_h_B6 = 46.672 #mm, slant length of horn, from Greer memo, and calculated from a_h and NRAO FEND-40.02.06.01-001-P-DWG
    R_h_B7 = 45.7817 # slant length of horn, from ALMA Band 7 cartridge Preliminary design report, p.29


    ###########################################################################################
    ####################  EDIT BELOW HERE #####################################################
    ################## user-provided system parameters  #######################################

    #mirror parameters that are common between bands (but not common with SPT, so we can change them)
    f_M3 = 70 # focal length of M3  
    f_M4 = 60.2   # focal length of M4 
    d_23 = 2208.21+130 # distance between M2 and M3
    d_34 =  220      # distance between M3 and M4 
    primary_truncate_dB = 29.93665 #This is set by the position and size of the existing primary and secondary.  If we could chooose, we'd choose 11 dB (see Goldsmith Fig 6.6 for motivation to choose 11 dB)


    #mirror parameters that are not common between bands
    lambda_b6 = [1.1, 1.25, 1.4] # mm, wavelengths to plot for Band 6
    f_M5_b6 = 170       # thin lens equivalent focal length of M5, band 6 (cold mirror), mm
    f_M6_b6 = 53.731
    d_45_b6 = 230+140          # distance between M4 and M5 for band 6, mm.
    d_56_b6 = 340        # distance between M5 and M6 for band 6, mm.
    d_6h_b6 = 57.940           #distance between M6 and horn aperture for band 6, mm

    lambda_b7 = [.8, .95, 1.1] # mm, band 7 wavelengths to plot
    f_M5_b7 = 65.990      # mm, thin lens equivalent focal length of M5, band 7 (cold mirror)
    d_45_b7 = 230+260      # mm, distance between M4 and M5 for band 7
    d_5h_b7 = 74.266       # mm, distance between M5 and horn aperture for band 7
    
    
    # Mirror radius, window spacing, and dichroic spacing parameters below only impact the plotting, not any of the calculations
    #                       M1 ,  M2, M3,  M4,  M5,  M6 
    mirror_max_radii_b6 = [5000, 875, 65,  33,  34,  29] #mm, maximum tolerable radius of mirrors.  Marked in plots with a vertical (5w) and horizontal (4w) black bar.  Does not affect beam propagation
    mirror_max_radii_b7 = [5000, 875, 65,  33,  38] #mm, maximum tolerable radius of mirrors.  Marked in plots with a vertical (5w) and horizontal (4w) black bar. Does not affect beam propagation
    dichroic_spacing = 230 #mm, distance from M4 to the dirchoic (Dichroic is marked on the plots with a dark blue line).  Does not affect beam propagation. 
    window_spacing_b6 = 230 + 140 + 128 # mm, distance from M4 to the 300 K window.  Window marked on plots with a cyan line.  Does not affect beam propagation.  
    window_spacing_b7 = 230 + 37.9995        # mm, distance from M4 to the front face of the vacuum vessel (not the window insert). Window marked on plots with a cyan line.  Does not affect beam propagation.  

    
    ######################################################################################
    ############################# EDIT ABOVE HERE ########################################
    ######################################################################################
    

 
    
    ############ put all the parameters into a dict  #################
    beams = {'B6':{}, 'B7':{}}

    beams['B6']['focal_lengths'] = [f1, f2, f_M3, f_M4, f_M5_b6, f_M6_b6]   
    beams['B6']['mirror_max_radii'] = mirror_max_radii_b6
    beams['B6']['element_spacings'] = [d_12, d_23, d_34, d_45_b6, d_56_b6, d_6h_b6]
    beams['B6']['dichroic_spacing'] = dichroic_spacing
    beams['B6']['window_spacing'] = window_spacing_b6
    
    beams['B7']['focal_lengths'] = [f1, f2, f_M3, f_M4, f_M5_b7]
    beams['B7']['mirror_max_radii'] = mirror_max_radii_b7
    beams['B7']['element_spacings'] = [d_12, d_23, d_34, d_45_b7, d_5h_b7]
    beams['B7']['dichroic_spacing'] = beams['B6']['dichroic_spacing']
    beams['B7']['window_spacing'] = window_spacing_b7
    


    ####### calculated system parameters  #########
    
    # band 6
    beams['B6']['lambda'] = lambda_b6
    beams['B6']['primary_truncate_dB'] = primary_truncate_dB
    beams['B6']['frequency'] = [ 1000*c/x for x in beams['B6']['lambda'] ]    #frequency in Hz
    beams['B6']['a_h'] = a_h_B6  # horn aperture radius
    beams['B6']['R_h'] = R_h_B6  # slant length of horn
    beams['B6']['w_h'] = beams['B6']['a_h']*0.644 # beam radius at the aperture of the horn.  # Goldsmith eq 7.41.  The .644 comes from fig 7.6.  Assumes corrugated circular horn.
    
    # band 7
    beams['B7']['lambda'] = lambda_b7
    beams['B7']['primary_truncate_dB'] = primary_truncate_dB
    beams['B7']['frequency'] = [ 1000*c/x for x in beams['B7']['lambda'] ] #frequency in Hz
    beams['B7']['a_h'] = a_h_B7  # horn aperture radius
    beams['B7']['R_h'] = R_h_B7  # slant length of horn
    beams['B7']['w_h'] = beams['B7']['a_h']*0.644 #beam radius at the aperture of the horn.  # Goldsmith eq 7.41.  The .644 comes from fig 7.6.  Assumes corrugated circular horn.        
    
    return beams
    
    
def main(directions = ['forward'], plot = True, do_adjust = False):
    '''
    
    Wrapper function to do the beam propagation and plotting. 
    
    
    Parameters:
    -----------
    directions: list of strings, allowed values of list elements are 'forward' and
        'reverse'.  List may be a single element.  Forward means starting from the
        sky and going towards the horn.  Reverse means starting from the horn and 
        going towards the sky.
    plot: bool, whether or not to make the plots
    do_adjust: bool, if True the code will calculate what the last mirror focal 
        length and distance from last mirror to the horn need to be to achieve 
        perfect matching.  Those calculated values will be used for plotting. 
        If False, the code will use the user-provided focal length and element 
        spacing, even if they are not optimal. Only has an effect when looking 
        at the 'forward' direction.  Not implemented for 'reverse'. 
    
    
    Returns:
    --------
    beams: dict, a dictionary containing all of the beam info
 
    '''
    
    beams = generate_parameter_dict()
    
    for direction in directions:
        print('\n Beginning {}\n'.format(direction))
    
        for band in beams:
            print('\n Beginning {}\n'.format(band))
            beams[band] = beam_propagation(beams[band], direction, do_adjust=do_adjust)
              
        if plot:
            print(matplotlib.get_backend())
            thin_lens_plot(beams, direction)
    
    
    return beams
    


def beam_propagation(beam, direction = 'forward', point_density_index = 100, do_adjust = True):
    '''
    Propagates a gaussian beam through space.  Calculates beam radius along the path of the 
    beam, locations of beam waists, etc.
    
    
    Parameters:
    ----------
    beam: dict, one band (ie band 6 or band 7) at a time, dict producted by generate_parameter_dict()
    direction: string, either "forward" or "reverse".  Forward means starting from 
        the primary and going towards the receiver.  Reverse means starting at the horn antenna in 
        the receiver and going towards the primary/sky.
    point_density_index: int or float, how many points along the optical axis should we calculate 
        beam radii for?  This number is a multiplicative factor in the calculation of how many points
        to use.
    do_adjust: bool, whether stick with the user-set numbers for the last mirror 
        position and focal length (False) or to adjust them so the matching to 
        the horn is perfect for the middle wavelength (True).  Only has an effect 
        when working in the 'forward' direction
        
        
    Returns:
    --------
    beam: dict, the same dict that was provided as an input parameter but now 
        with additional keys added that capture info about the beam width along the z-axis, 
        locations and sizes of beam waists, etc
      
    
    '''
    
    print('do adjust: {}'.format(do_adjust))

    focal_lengths = beam['focal_lengths'].copy()
    element_spacings = beam['element_spacings'].copy()
    lambdas = beam['lambda']
    R_h = beam['R_h']
    w_h = beam['w_h']
    primary_truncate_dB = beam['primary_truncate_dB']  #TODO: actually, this will be set by the secondary size.  Calculate it.
    
    n_mirrors = len(focal_lengths) -1 # -1 because we stop/start at the prime focus rather than the primary
    
    if direction == 'reverse':
        focal_lengths.reverse()
        element_spacings.reverse()
    
    for y in range(len(lambdas)):  #cycle through the frequencies for this band
        lambd = lambdas[y]
    
        #calculate starting waist size and position
        if direction == 'forward': #starting waist is prime focus
            w_primary = 5000/(0.3393 * (primary_truncate_dB**.5))  #goldsmith 2.35b, beamwidth at primary
            w0_primary = min(w0_calc(lambd, w_primary, focal_lengths[0])) #primary beam waist radius      
            w0 = [w0_primary]  #start the list of beam waists
            z_w0 = [-(element_spacings[0]-focal_lengths[0])]  #z = 0 at secondary.  To make z = 0 at primary, set to focal_lengths[0]           
        if direction == 'reverse': #starting waist is horn waist
            w0_horn, z_offset_horn = w0z0(lambd, R_h, w_h) #z_offset is the distance inside the horn (measured from the aperture) where the beam waist falls
            if 'horn_offset' in beam: #if we have already started the horn offset list for the different frequencies
                beam['horn_offset'].append(z_offset_horn)
            else: 
                beam['horn_offset'] = [z_offset_horn]
            w0 = [w0_horn]
            z_w0 = [-z_offset_horn]

              
        #calculate mirror positions
        for m in range(n_mirrors):
            if m == 0: #first one
                if direction == 'forward':
                    mirror_z = [0]  #forward:  z = 0 at secondary
                elif direction == 'reverse':
                    mirror_z = [element_spacings[0]]  #reverse: z = 0 at horn aperture
                else:
                    raise ValueError('direction argument must be forward or reverse')
            else:
                new_z = mirror_z[-1] + element_spacings[m]
                mirror_z.append(new_z)
                 
        if direction == 'forward':
            d_starting_waist_to_next_lens = mirror_z[0]-z_w0[0]
        elif direction == 'reverse':
            d_starting_waist_to_next_lens = element_spacings[0] + z_offset_horn #treating element_spacings as measuring from the horn aperture
            
        
          
        #calculate waist radii and positions    
        for m in range(n_mirrors): 
            if m == 0: #it's the first one, use the starting point stuff from above
                d_in = d_starting_waist_to_next_lens  
                previous_w0 = w0[0]
            else: #not the first one
                d_in = mirror_z[m] - previous_z_w0
            if direction == 'forward':
                w0_out, d_out = lens(previous_w0, d_in, focal_lengths[m+1], lambd)  #skip primary
            elif direction == 'reverse':
                w0_out, d_out = lens(previous_w0, d_in, focal_lengths[m], lambd)
            w0.append(w0_out)
            z_w0.append(mirror_z[m] + d_out)
            previous_w0 = w0[-1]
            previous_z_w0 = z_w0[-1]
            
        #calculate beam radius of curvature at mirror and 5w mirror diameters
        R_mirror = [] #beam radius of curvature going into the mirror (i.e. on the sky side of the mirror if direction=='forward' and on the horn side of the mirror if direction == 'reverse')
        if lambd == np.max(lambdas):
            mirror_diameters = []
        if direction == 'forward':  ##TODO: get the indexing to work for reverse too
            for m in range(n_mirrors+1):
                if m == 0: #primary mirror
                    curvature = 0
                    w5 = 10000
                else:
                    d_in = mirror_z[m-1] - z_w0[m-1]
                    #print('d_in:  {} - {} = {}'.format(mirror_z[m-1], z_w0[m-1], d_in))
                    #print('w0: ', w0[m-1], '\n')
                    width, curvature = wR(lambd, d_in, w0[m-1])
                    w5 = width*5
                R_mirror.append(curvature)
                if lambd == np.max(lambdas):
                    mirror_diameters.append(w5)
        
            
        if direction == 'forward' and y == 1 and do_adjust == True: #try to make the last mirror before the horn have the correct focal length to match the horn at the middle frequency
            
            adjust = True
            #figure out what the mirror f should be and what the distence between the last mirror and the horn should be
            w0_in = w0[-2]
            w0_horn, z_offset_horn = w0z0(lambd, R_h, w_h) #z_offset is the distance inside the horn (measured from the aperture) where the beam waist falls
            d_in = mirror_z[-1] - z_w0[-2]
            print('d_in, w0_in, w0_horn', d_in, w0_in, w0_horn)
            tmp1, tmp2, f_pos_w, f_neg_w = reverse_lens(w0_in, d_in, 100, w0_horn, lambd)          

            if np.isreal(f_pos_w) and np.isreal(f_pos_w):  #both are real
                if f_pos_w > 0 and f_neg_w > 0: #both positive.  Warn and take smaller one.
                    wa.warn('Two positive roots for last mirror focal length: {}, {}.  Taking smaller one.'.format(f_pos_w, f_neg_w))
                    focal_length = min([f_pos_w, f_neg_w])
                elif f_pos_w > 0: #this is the usable root
                    focal_length = f_pos_w
                elif f_neg_w > 0: #this is the usable root
                    focal_length = f_neg_w
                else: #no positive roots
                    wa.warn('no positive roots for last mirror focal length to match horn')
                    adjust = False
            elif np.isreal(f_pos_w) and f_pos_w > 0: #one real root and it's positive
                focal_length = f_pos_w
            elif np.isreal(f_neg_w) and f_neg_w > 0: #one real root and it's positive
                focal_length = f_neg_w
            else:  #no real positive roots
                wa.warn('no real positive roots for last mirror focal length to match horn')
                adjust = False
                
            if adjust:  #we found a real positive solution for the focal length.  Now apply it
            
                #fix the current wavelength                
                w0_out, d_out = lens(w0_in, d_in, focal_length, lambd)
                print('w0_out, w0_horn: {}, {}'.format(w0_out, w0_horn))
                new_element_spacing = d_out - z_offset_horn
                print('adjusting last mirror focal length to match horn. f = {}mm'.format(focal_length))
                print('adjusting space from last mirror to horn aperture: d = {}mm'.format(new_element_spacing))
                w0[-1] = w0_out
                z_w0[-1] = mirror_z[-1] + d_out
                element_spacings[-1] = new_element_spacing
                focal_lengths[-1] = focal_length
                
                #fix the previous lambda
                w0_in = w0_matrix[0][-2]
                d_in = mirror_z[-1] - z_w0_matrix[0][-2]
                w0_out, d_out = lens(w0_in, d_in, focal_length, lambdas[0])
                w0_matrix[0][-1] = w0_out
                z_w0_matrix[0][-1] = mirror_z[-1] + d_out
                
                #store
                beam['element_spacings'] = element_spacings
                beam['focal_lengths'] = focal_lengths
                
                
                
        #store new calculated lists into the per-frequency matrices      
        if y == 0: #we haven't made the per-frequency matrices yet
            w0_matrix = [w0]
            z_w0_matrix = [z_w0]
            R_mirror_matrix = [R_mirror]
        else:
            w0_matrix.append(w0)
            z_w0_matrix.append(z_w0)
            R_mirror_matrix.append(R_mirror)

         
    #now fill in all the points of the beam envelope all along the z-axis        
    for y in range(len(lambdas)):  #cycle through the frequencies for this band
        lambd = lambdas[y]
        
        w0 = w0_matrix[y]
        z_w0 = z_w0_matrix[y]
                
        #generate points along z-axis
        z_list = []
        for z in range(len(z_w0)-1):
            distance = np.abs(z_w0[z+1] - z_w0[z]) #distance from one waist to the next
            n_points = int(np.ceil(distance*point_density_index/np.log(distance)))
            z_steps = np.linspace(z_w0[z], z_w0[z+1], n_points)
            if z == 0: #first one
                z_list.extend(z_steps)
            else: 
                z_list.extend(z_steps[1:]) #the first point is already there from the previous round
                               
        #for every point in z_list, calculate the beam radius (w), radius of wavefront curvature (R), phase slippage (phi)
        w = []
        R = []
        phi = []
        waist_counter = 0
        if direction == 'forward':
            current_mirror_z = mirror_z[0]
            for z in z_list:
                if z > current_mirror_z and waist_counter<len(w0)-1: #we have reached a mirror, move to the next waist
                    waist_counter += 1
                    if waist_counter< len(mirror_z):
                        current_mirror_z = mirror_z[waist_counter]
                z_offset = z-z_w0[waist_counter]
                w_tmp, R_tmp = wR(lambd, z_offset, w0[waist_counter])
                phi_tmp = phi_slippage(lambd, z_offset, w0[waist_counter])
                w.append(w_tmp)
                R.append(R_tmp)
                phi.append(phi_tmp)
        elif direction == 'reverse':
            current_mirror_z = mirror_z[0]
            for z in z_list:
                if z > current_mirror_z and waist_counter<len(w0)-1: #we have reached a mirror, move to the next waist
                    waist_counter += 1
                    if waist_counter < len(mirror_z):
                        current_mirror_z = mirror_z[waist_counter]
                z_offset = z-z_w0[waist_counter]
                w_tmp, R_tmp = wR(lambd, z_offset, w0[waist_counter])
                phi_tmp = phi_slippage(lambd, z_offset, w0[waist_counter])
                w.append(w_tmp)
                R.append(R_tmp)
                phi.append(phi_tmp)
                      
        #store new calculated lists into the per-frequency matrices      
        if y == 0: #we haven't made the per-frequency matrices yet
            w_matrix = [w]
            R_matrix = [R]
            z_list_matrix = [z_list]
            phi_matrix = [phi]
        else:
            w_matrix.append(w)
            R_matrix.append(R)
            z_list_matrix.append(z_list)
            phi_matrix.append(phi)
            
    
    #calculate the waist offsets at the horn
    z_w0_horn_beam_offsets = []
    w0_horn_beam_offsets = []
    for i in range(len(lambdas)):
        lambd = lambdas[i]
        w0_beam = w0_matrix[i][-1]
        z_w0_beam = z_w0_matrix[i][-1]
        
        R_h = beam['R_h']
        w_h = beam['w_h']
        horn_aperture_position = mirror_z[-1] + beam['element_spacings'][-1]
        w0_horn, z_offset_horn = w0z0(lambd, R_h, w_h) #z_offset is the distance inside the horn from the aperture where w0 falls
        z_w0_horn = z_offset_horn + horn_aperture_position
        
        w0_diff = w0_horn - w0_beam
        z_w0_diff = z_w0_horn - z_w0_beam
        w0_horn_beam_offsets.append(w0_diff)
        z_w0_horn_beam_offsets.append(z_w0_diff)
        
    #print('Horn-beam wiast offsets: {}'.format(np.round(w0_horn_beam_offsets,3)))
    #print('Horn-beam z_w0  offsets: {}'.format(np.round(z_w0_horn_beam_offsets,3)))
    
    
    
    #calculate mirror shape
    mirror_shape = []
    if direction == 'forward':
        for x in range(len(R_mirror_matrix[1])):
            R1 = R_mirror_matrix[1][x]
            if R1 == 0:
                R2 = 0
            else:
                R2 = (focal_lengths[x]*R1)/(R1-focal_lengths[x])
            mirror_shape.append('{}, {}'.format(np.round(R1, 1), np.round(R2,1)))
        print('\nMirror shapes: {}'.format(mirror_shape[2:]))
        print('mirror diameters: {}\n'.format(mirror_diameters[2:]))
    
    #store everything in the beam dictionary
    beam['w0'] = w0_matrix
    beam['z_w0'] = z_w0_matrix
    beam['w'] = w_matrix
    beam['R'] = R_matrix
    beam['R_mirror'] = R_mirror_matrix
    beam['mirror_shapes'] = mirror_shape
    beam['mirror_diameters'] = mirror_diameters    
    beam['mirror_z'] = mirror_z
    beam['z_list'] = z_list_matrix
    beam['phi'] = phi_matrix
    beam['horn_beam_w0_offsets'] = w0_horn_beam_offsets
    beam['horn_beam_z_offsets'] = z_w0_horn_beam_offsets

        
    return beam
    
   
    
def thin_lens_plot(beams, direction = 'forward', phase = False, R = True, single_band = False):
    '''
    
    Parameters:
    ----------
    beams: dict, a dictionary containing all of the beam info
    direction: string, either "forward" or "reverse".  Forward means starting from 
        the primary and going towards the receiver.  Reverse means starting at the horn antenna in 
        the receiver and going towards the primary/sky.
    phase: bool, whether or not to plot the phase slippage
    R: bool, whether or not to plot the radius of curvature of the beam
    single_band: bool or string, may be False or 'B6' or 'B7'.  Use if your 'beams' 
        variable contains data for two bands but you only want to actually plot 
        one of the bands.  
        
        
    Returns:
    --------
    none, but makes plots
   
    '''
    
    
    plt.ion()
    color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    
    for band in beams:
        if single_band:
            if single_band == band:
                beam = beams[band]
            else:
                continue
        else:
            beam = beams[band]
        R_h = beam['R_h']
        w_h = beam['w_h']
        plt.figure()
        for row in range(len(beam['lambda'])): #each frequency within B6 or B7 to look at
            label = str(np.round(beam['lambda'][row], 2)) + ' mm'
            lambd = beam['lambda'][row]
            plt.plot(beam['z_list'][row], beam['w'][row], '.', markersize = 3, label = label, color = color_list[row])
            
            #horn waist position
            if direction == 'forward':
                horn_aperture_position = beam['mirror_z'][-1] + beam['element_spacings'][-1]
                w0_horn, z_offset_horn = w0z0(lambd, R_h, w_h) #z_offset is the distance inside the horn from the aperture where w0 falls
                plt.plot(horn_aperture_position + z_offset_horn, w0_horn, '*', color = color_list[row])
                
            
            if phase:
                plt.plot(beam['z_list'][row], np.asarray(beam['phi'][row])*100, ':', linewidth = 1, color = color_list[row], alpha = 0.5)
            
            if R:
                plt.plot(beam['z_list'][row], np.asarray(beam['R'][row])/10, '-.', linewidth = 1, color = color_list[row], alpha = 0.5)
            
            
            mirror_max_radii = beam['mirror_max_radii'].copy() #.copy avoids python mutability nonsense
            if direction == 'reverse':
                mirror_max_radii.reverse()
            #print(mirror_max_radii)
            #print(beam['mirror_z'])
            
            #vertical lines and labels for mirrors, dichroic, and windows
            if direction == 'forward':
                for x in range(len(beam['mirror_z'])):
                    plt.vlines(beam['mirror_z'][x], ymin = 0, ymax = mirror_max_radii[x+1]/2.5, color = 'black') #5w
                    plt.hlines(mirror_max_radii[x+1]/2, xmin = beam['mirror_z'][x]-10, xmax = beam['mirror_z'][x]+10, color = 'black') #4w
                    plt.text(beam['mirror_z'][x]-20, -5, 'M{}'.format(x+2))
                dichroic_z = beam['mirror_z'][2] + beam['dichroic_spacing']
                plt.text(dichroic_z-20, -5, 'D')
                window_z = beam['mirror_z'][2] + beam['window_spacing']
                plt.text(window_z-1, -5, 'W')
                plt.vlines(dichroic_z, ymin = 0, ymax = 100/5, color = 'blue')
                plt.vlines(window_z, ymin = 0, ymax = 50/5, color = 'cyan') #front face ov vac shell
                plt.vlines(window_z+17.272, ymin = 0, ymax = 50/5, color = 'cyan', alpha = .8)  #inside face of vac shell
                plt.vlines(window_z+31.242, ymin = 0, ymax = 50/5, color = 'cyan')  #outside of 50k shell
                plt.vlines(window_z+32.8295, ymin=0, ymax = 50/5, color = 'cyan', alpha = .8) #inside of 50k shell
            elif direction == 'reverse':
                for x in range(len(beam['mirror_z'])):
                    plt.vlines(beam['mirror_z'][x], ymin = 0, ymax = mirror_max_radii[x]/2.5, color = 'black') #5w
                    plt.hlines(mirror_max_radii[x]/2, xmin = beam['mirror_z'][x]-10, xmax = beam['mirror_z'][x]+10, color = 'black') #4w
                    plt.text(beam['mirror_z'][x]-20, -10, 'M{}'.format(len(beam['mirror_z'])-x+1))
                 
            # vertical lines for beam waists
            first_waist = 1
            for y in range(len(beam['w0'][row])):
                if first_waist == 1:
                    plt.vlines(beam['z_w0'][row][y], ymin = 0, ymax = 10, color = color_list[row], alpha = 0.3, label = label + ' waist')
                    first_waist = 0
                else:
                    plt.vlines(beam['z_w0'][row][y], ymin = 0, ymax = 10, color = color_list[row], alpha = 0.3)
       
       
        plt.grid()
        #plt.xlim([1800,3600])
        plt.ylim([-25,450])
        plt.xlabel('distance along optical axis [mm, secondary at 0]', fontsize = 14)
        plt.ylabel('1/e beam radius [mm]', fontsize=14)
        plt.title('{} {}'.format(band, direction), fontsize=14)
        plt.legend(loc='upper right')
        #plt.tight_layout()
        plt.pause(0.001) #event pump to keep plt from freezing
                
        


def w0z0(lambd, R, w):
    '''
    Calculate the beam waist radius and location when you know the radius of 
    curvature (R) at some other particular beam radius (w). 
    
    Parameters:
    -----------
    lambd: float, wavelength in mm
    R: float or int, the radius of curvature in mm at the location with beam radius w
    w: float or int, the beam radius in mm at the location with radius of curvature R
    
    Returns:
    --------
    w0 float, the beam diameter in mm at the beam waist
    z0: float, the offset along the z axis in mm between the beam waist and the 
        location with beam radius w
    
    
    '''
    
    w0 = w/((1+(np.pi*w**2/(lambd*R))**2)**.5)  # goldsmith table 2.3 line 6
    z0 = R/(1+(lambd*R/(np.pi*w**2))**2)        # goldsmith table 2.3 line 6.  Note I have taken out a minus sign compared to the Lingzhen Zeng code, to match Goldsmith and treat this as a non-directed offset
    
    return w0, z0


def wR(lambd, z, w_0): 
    '''
    Get the beam radius (w) and radius of curvature (R) 
    when you have the beam waist (w_0), propagation distance (z), and wavelength (lambd).
    Typically use mm for the lengths, but this function on its own works for any consistent unit. 

    Parameters:
    ----------
    lambd: int or float, wavelength (lambd instead of lambda because lambda is a special word in python)
    z: int or float, distance along the optical axis from the beam waist
    w_0: int or float, the radius of the beam waist

    Returns:
    --------
    w: int or float, the 1/e Gaussian beam radius at location z
    R: int or float, the radius of curvature of the wavefront at location z
    
    '''
    w = float(w_0*np.sqrt(1+(lambd*z/(3.14159*w_0**2))**2))    #goldsmith eq 2.21b and 2.26c
    if z == 0:
        R = float('nan')
    else:
    
        try:
            R = z*(1+(3.14159*w_0**2/(lambd*z))**2)             #goldsmith eq 2.21a and 2.26b
        except:
            print('exception1')
            if z == 0:
                R = float('nan')
            else:
                raise
    
    return w, R
    
    
def phi_slippage(lambd, z, w0):
    '''
    Calculate the phase slippage when you know the beam waist radius, z offset from the waist, and wavelength
    
    
    Parameters:
    -----------
    lambd: float or int, wavelength typically in mm
    z: float or int, distance along the optical axis from the beam waist, typically in mm
    w0: float or int, radius of the beam waist, typically in mm
    
    Returns:
    -------
    phi: float, Gaussian beam phase shift in radians
    
    
    '''
    
    phi = np.arctan((lambd*z)/(np.pi*w0**2))    # Goldsmith 2.26d
    
    return phi



def w0_calc(lambd, w, z):
    '''
    Calculate the beam waist radius, w0, given the beam radius at a given distance, z, 
    away from the beam waist.  
    
    Parameters:
    ----------
    lambd: float, wavelength typically in mm
    w: float, 1/e beam width radius at a distance z from the beam waist, typically in mm
    z: float, distance from the beam waist to where w is measured, typically in mm
    
    '''
    #from Goldsmith table 2.3 line 3
    w0_pos = (((w**2)/2) * (1 + ((1 - ((2*lambd*z/(3.1415*(w**2)))**2))**.5)))**.5
    w0_neg = (((w**2)/2) * (1 - ((1 - ((2*lambd*z/(3.1415*(w**2)))**2))**.5)))**.5
    
    return w0_pos, w0_neg


def lens(w0_in, d_in, f, lambd):
    '''
    Calculates the beam transformation through a paraxial thin lens (or equivalent mirror).
    Note this function by itself works with any consistent units, but in typical usage 
    within this module all of the parameters are in mm, so that's why the Parameters
    and Returns descriptions specify those units. 
    
    Parameters:
    ----------
    w0_in: float, the beam waist radius in mm, before the lens/mirror.  See 
        Goldsmith fig 3.6
    d_in: float, the distance between the lens/mirror and w0_in, along the z axis, in mm.
        See Goldsmith fig 3.6
    f: float, the focal length of the lens/mirror in mm
    lambd: float, the wavelength in mm
    
    
    Returns:
    --------
    w0_out: float, the beam waist radius in mm, after the mirror/lens.  See 
        Goldsmith fig 3.6
    d_out: float, the distance between the lens/mirror and w0_out in mm.  See
        Goldsmith fig 3.6 
        
    '''
    
    z = 3.14159*(w0_in**2)/lambd
    c = d_in/f
    
    d_out = (1 + ((c-1) / (((c-1)**2) + ((z**2)/(f**2))))) * f      # Goldsmith 3.31a
    w0_out = w0_in / ((((c-1)**2) + ((z**2)/(f**2)))**.5)           # Goldsmith 3.31b
    
    if d_out < 0:
        print('WARNING: d_out < 0: d_out = {} for f = {}'.format(d_out, f))
    
    return w0_out, d_out
    
    

def reverse_lens(w0_in, d_in, d_out, w0_out, lambd):
    '''
    Solve for the needed focal length given the desired d_out and d_in.
    OR
    Solve for the needed focal length given the desired w0_in and w0_out.
    
    Uses Goldsmith equations 3.31a and 3.31b, solved for f
    
    Parameters:
    -----------
    w0_in: float, desired input beam waist
    w0_out: float, desired output beam wasit
    d_in: float, desired input distance from waist to lens
    d_out: float, desired output distance from lens to waist
    lambd: float, the wavelength in mm
    
    
    Returns:
    --------
    f_pos_d: float, needed focal length using the requested d_in and d_out (but 
        not the requested waists). Positive root.  
    f_neg_d: float, needed focal length using the requested d_in and d_out (but 
        not the requested waists). Negative root.  
    f_pos_w: float, needed focal length using the requested w0_in and w0_out (but 
        not the requested distances). Positive root.  
    f_neg_w: float, needed focal length using the requested w0_in and w0_out (but 
        not the requested distances). Negative root.
    
    
    
    '''
    

    zc = 3.14159*(w0_in**2)/lambd
    
    #using the d_out equation (given d_in, d_out, w0_in.  W0_out and f free)
    A = d_out + d_in
    B = (-2*d_in*d_out) - (d_in**2) - (zc**2)
    C = ((d_in**2) * d_out) + (d_out * (zc**2))  # plus or minus sign here is questionable
    
    
    f_pos_d = (-B + ((B**2 - (4*A*C))**.5)) / (2*A)
    f_neg_d = (-B - ((B**2 - (4*A*C))**.5)) / (2*A)
    
    
    #using the w0_out equation.  Given w0_in, w0_out, d_in.  d_out, and f free.  
    A = 1 - ((w0_in/w0_out)**2)
    B = -2*d_in
    C = (zc**2) + (d_in**2)
    
    f_pos_w = (-B + ((B**2 - (4*A*C))**.5)) / (2*A)
    f_neg_w = (-B - ((B**2 - (4*A*C))**.5)) / (2*A)
    
    [w0_out_neg, d_out] = lens(w0_in, d_in, f_neg_w, lambd)
    [w0_out_pos, d_out] = lens(w0_in, d_in, f_pos_w, lambd)
    #print('M5 to horn waist dist for waist matching: {}, {}'.format(np.round(w0_out_pos, 2), np.round(w0_out_neg, 2)))
       
    return f_pos_d, f_neg_d, f_pos_w, f_neg_w
    
def lens_explorer(w0_in, d_in, f_list, lambd):
    '''
    Makes a plot of focal lengths vs d_outs for a given d_in and w0_in
    
    '''
    plt.ion()
    d_out = []
    w0_out = []
    for f in f_list:
        w0_out_tmp, d_out_tmp = lens(w0_in, d_in, f, lambd)
        d_out.append(d_out_tmp)
        w0_out.append(w0_out_tmp*10)
        
    ind_max_dout = np.argmax(d_out)
    f_max_dout = f_list[ind_max_dout] #what focal length gives the largest d_out
    print('Focal length for largest d_out: {} mm'.format(np.round(f_max_dout,4)))
    print('d_out at that focal length:     {} mm'.format(np.round(d_out[ind_max_dout],4)))
    
    plt.figure()
    plt.plot(f_list, d_out, label = 'd_out')
    plt.grid()
    plt.xlabel('focal length [mm]')
    plt.tight_layout()
    plt.plot(f_list, w0_out, label = 'w0')
    plt.legend()
    plt.ylabel('w0_out*10 or d_out [mm]')
    plt.tight_layout()
    plt.pause(0.001) #event pump to keep plt from freezing
        
def M5_focal_vs_mirror_size(lambd, w0_in, d_out):
    '''
    Makes a plot showing the M5 focal length required for the mirror closest to the horn
    vs the required 5w mirror diameter for a given d_out (which should be roughly the distance
    from the mirror to the window)
    
    '''           
    plt.figure()
    pos = []
    neg = []
    diameter = []
    d_in_list = np.linspace(.1,150,3000)
    for d_in in d_in_list:
        w_tmp, R_tmp = wR(lambd, d_in, w0_in)
        diameter.append(w_tmp*2*5/10) #cm, dimeter of 5w mirror
        a,b,c,d = reverse_lens(w0_in, d_in, d_out, 1, lambd)
        pos.append(a)
        neg.append(b)
    plt.plot(diameter, pos, label = 'positive root')
    plt.plot(diameter, neg, label = 'negative root')
    plt.grid()
    plt.legend()
    plt.xlabel('5w mirror diameter [cm]')
    plt.ylabel('focal length of M5 [mm]')
    plt.title('Requirement for d_out of {}'.format(d_out))
    
    plt.figure()
    plt.plot(d_in_list, pos, label = 'positive root')
    plt.plot(d_in_list, neg, label = 'negative root')
    plt.grid()
    plt.legend()
    plt.xlabel('d_in [mm]')
    plt.ylabel('focal length of M5 [mm]')
    plt.title('Requirement for d_out of {}'.format(d_out))
    plt.pause(0.001) #event pump to keep plt from freezing
    
    
    indices = np.where(~np.isnan(neg))
    min_5w = diameter[indices[0][0]]
    min_5w_inches = min_5w/2.54
    min_focal_neg = neg[indices[0][0]]
    min_focal_pos = pos[indices[0][0]]
    d_in_min = d_in_list[indices[0][0]]
    
    print('Minimum 5w diameter: {} cm, {} in.'.format(np.round(min_5w, 2), np.round(min_5w_inches, 2)))
    print('focal lengths: {} mm (pos), {} mm (neg).'.format(np.round(min_focal_pos,2), np.round(min_focal_neg, 2)))
    print('d_in {} mm'.format(np.round(d_in_min, 2)))


    #return(diameter, neg)
    
def horn_coupling_calculator(beam):
    '''
    Assumes the two beams are perfectly co-axial, and the only mismatch is coming
    from the beam waist (i.e. the beam coming in from the sky) being a slightly different 
    size and position than the horn waist (i.e. the beam coming from the horn antenna).  
    Assumes a well-behaved corrugated conical horn antenna in the style of Goldsmith 
    table 7.1 and section 7.6.2. 
    
    '''
    eff = []
    lambdas = beam['lambda']
    R_h = beam['R_h']
    w_h = beam['w_h']
    
    for i in range(len(lambdas)):
        lambd = lambdas[i]
        beam_waist = beam['w0'][i][-1]
        beam_waist_position = beam['z_w0'][i][-1]
        horn_aperture_position = beam['mirror_z'][-1] + beam['element_spacings'][-1]
        w0_horn, z_offset_horn = w0z0(lambd, R_h, w_h) #z_offset is the distance inside the horn from the aperture where w0 falls
        horn_waist_position = horn_aperture_position + z_offset_horn
        waist_spacing = abs(horn_waist_position - beam_waist_position)
        
        #calculate horn beam width and radius of curvature at location of beam waist.  
        w_horn, R_horn = wR(lambd, waist_spacing, w0_horn)
        
        coupling_efficiency = 4/((((beam_waist/w_horn) + (w_horn/beam_waist))**2) + (((3.14159*beam_waist*w_horn)/(lambd*R_horn))**2))
        eff.append(coupling_efficiency)
        
    return eff

def tilt_coupling_calculator(theta_degrees, w0, lambd):
    '''
    Coupling efficiency of tilted beams.  Assumes the beams are perfectly matched 
    (same waist size and location) except that one beam is tilted by a small angle
    theta.  Assumes the meeting point of the beams is right at the waist.
    
    w0 and lambd arguments must be in the same units (usually mm)
    
    tilt coupling is better for smaller beam waists
    '''
    
    theta = np.deg2rad(theta_degrees)
    
    eff = 100 * np.exp(-(3.1415**2)*(w0**2)*(theta**2)/(lambd**2))   #from Goldsmith 4.24a and 4.25 with matched waists
    
    return np.round(eff,3)
    

def offset_coupling_calculator(x0, w01, w02, lambd, z0=0):
    '''
    Assumes perfectly matched beams except for a lateral offset (see Goldsmith 
    figure 4.1c... the offset here is x0 in that diagram).  Assumes beams meet 
    at the beam waist.  All input parameters must be in the same units (usually mm).
    
    Parameters:
    -----------
    x0: float, lateral offset of the beam (see Goldsmith fig 4.1c)
    w01: float, waist radius of beam 1
    w02: float, waist radius of beam 2
    lambd: float, wavelength 
    z0: float, axial offset of waists (see delta-z in Goldsmith fig 4.1a)
    
    
    nb: 0.1 mm is about 4 mils
        offset coupling is better for larger beam waists
    
    '''
    
    delta_off = (((((w01**2) + (w02**2))**2) + ((lambd*z0/3.14159)**2)) / ((w01**2) + (w02**2)))**.5  #Goldsmith 4.30b
    
    eff = np.exp(-2*(x0**2)/(delta_off**2)) #goldsmith 4.30a
    
    
    #eff = 100*(np.exp(-(offset**2)/(w0**2)))  #from Goldsmith 4.30a and 4.31 with waists equal and no z offset

    return np.round(eff*100, 3)




