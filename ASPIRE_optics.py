

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
    2) run either main()





#TODO: - Use a config file to set mirror params instead of hard-coding in generate_parameter_dict()
       - General cleanup, block comments



'''



import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def generate_parameter_dict():
    '''
        ##ALMA band 6: 211-275 GHz.  Band 7: 275-373.  Band 2: 67-116.
    
    '''
    
    #physics parameters and unchangable telescope and horn parameters
    c = 299792458 #m/s, speed of light in a vacuum
    f1 = 8003.21 #mm, focal length of the primary, from Padin 2008
    f2 = 818.2808 #mm, effective (thin lens equiv) focal length of the secondary, from Stark memo 2018.  f1 = 1310.0 mm, f2 = 2180.0 mm
    d_12 = 8003.21 + 1310 # mm, distance between the primary and secondary, from Stark memo. Focal length + distance from prime focus to M2
    dg = 2180 # mm, distance from secondary to gregorian focus.  From Stark memo.
    a_h_B6 = 3.54 #mm, horn aperture radius, from Greer memo and confirmed by personal communication with John Effland
    a_h_B7 = 3.00225 # horn aperture radius, from ALMA Band 7 Cartridge Preliminary design report p.29
    R_h_B6 = 46.672 #mm, slant length of horn, from Greer memo, and calculated from a_h and NRAO FEND-40.02.06.01-001-P-DWG
    R_h_B7 = 45.7817 # slant length of horn, from ALMA Band 7 cartridge Preliminary design report, p.29


    ######################################################################################
    ############# user-provided system parameters  #######################################

    #mirror parameters that are common between bands (but not common with SPT, so we can change them)
    f_M3 = 103#115#111.25  # focal length of M3  #135 in Gene's design
    f_M4 = 130#130# 133.75  # focal length of M4 #151 in Gene's
    d_23 = dg+135 #160   # distance between M2 and M3
    d_34 =  600 #570      # distance between M3 and M4 #600 in Gene's design
    primary_truncate_dB = 11 #dB truncation of beam at primary (see Goldsmith Fig 6.6 for motivation to choose 11 dB)


    #mirror parameters that are not common between bands
    lambda_b6 = [1.1, 1.32, 1.4] #[214.1e9, 227.1e9, 272.5e9]   # band 6 frequency, Hz,  1.4 mm, 1.32 mm, 1.1 mm
    f_M5_b6 = 80 #72.6244        # thin lens equivalent focal length of M5, band 6 (cold mirror), mm
    f_M6_b6 = 25

    d_45_b6 = 900 #995          # distance between M4 and M5 for band 6, mm.  918.903 in gene's
    d_56_b6 = 62.122 #60        # distance between M5 and M6 for band 6, mm.
    d_6h_b6 = 40                #distance between M6 and horn aperture for band 6, mm

    lambda_b7 = [.8, .86, 1.1] #[272.5e9, 348.6e9, 374.7e9]    # band 7 frequency, Hz, 1.1 mm, .86 mm, .8 mm
    f_M5_b7 = 34.5      # thin lens equivalent focal length of M5, band 7 (cold mirror), mm
    f_M6_b7 = 25

    d_45_b7 = 900      # distance between M4 and M5 for band 7, mm
    d_56_b7 = 80       # distance between M5 and M6 for band 5, mm
    d_6h_b7 = 40       # distance between M6 and horn aperture for band 7, mm
    
    mirror_max_radii_b6 = [5000, 875, 178, 127, 80, 80]
    mirror_max_radii_b7 = [5000, 875, 178, 127, 80, 80]

    
    
    ######################################################################################
    ######################################################################################
    
    
    
    
    ############ put all the parameters into a dict  #################
    beams = {'B6':{}, 'B7':{}}

    beams['B6']['focal_lengths'] = [f1, f2, f_M3, f_M4, f_M5_b6, f_M6_b6]
    beams['B6']['mirror_max_radii'] = mirror_max_radii_b6
    beams['B6']['element_spacings'] = [d_12, d_23, d_34, d_45_b6, d_56_b6, d_6h_b6]  

    beams['B7']['focal_lengths'] = [f1, f2, f_M3, f_M4, f_M5_b7, f_M6_b7]  
    beams['B7']['mirror_max_radii'] = mirror_max_radii_b7
    beams['B7']['element_spacings'] = [d_12, d_23, d_34, d_45_b7, d_56_b7, d_6h_b7]   


    # calculated system parameters  #

    # band 6
    beams['B6']['lambda'] = lambda_b6
    beams['B6']['greg_foc'] = dg
    beams['B6']['primary_truncate_dB'] = primary_truncate_dB
    beams['B6']['frequency'] = [ 1000*c/x for x in beams['B6']['lambda'] ]    #frequency in Hz
    beams['B6']['a_h'] = a_h_B6  # horn aperture radius
    beams['B6']['R_h'] = R_h_B6  # slant length of horn
    beams['B6']['w_h'] = beams['B6']['a_h']*0.644 # beam radius at the aperture of the horn.  # Goldsmith eq 7.41.  The .644 comes from fig 7.6.  Assumes corrugated circular horn.
    
    # band 7
    beams['B7']['lambda'] = lambda_b7
    beams['B7']['greg_foc'] = dg
    beams['B7']['primary_truncate_dB'] = primary_truncate_dB
    beams['B7']['frequency'] = [ 1000*c/x for x in beams['B7']['lambda'] ] #frequency in Hz
    beams['B7']['a_h'] = a_h_B7  # horn aperture radius
    beams['B7']['R_h'] = R_h_B7  # slant length of horn
    beams['B7']['w_h'] = beams['B7']['a_h']*0.644 #beam radius at the aperture of the horn.  # Goldsmith eq 7.41.  The .644 comes from fig 7.6.  Assumes corrugated circular horn.        
    
    return beams
    
    
def main(direction = 'forward', plot = True):
    '''
 
    '''

    beams = generate_parameter_dict()
    
    for band in beams:
        beams[band] = beam_propagation(beams[band], direction)
        
    
    if plot:
        thin_lens_plot(beams, direction)
    
    
    return beams
    


def beam_propagation(beam, direction = 'forward', point_density_index = 10):
    '''
    
    '''

    focal_lengths = beam['focal_lengths']
    element_spacings = beam['element_spacings']
    lambdas = beam['lambda']
    R_h = beam['R_h']
    w_h = beam['w_h']
    primary_truncate_dB = beam['primary_truncate_dB']
    
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
            w0 = [w0_primary]
            z_w0 = [-(element_spacings[0]-focal_lengths[0])]  #z = 0 at secondary.  To make z = 0 at primary, set to focal_lengths[0]           
        if direction == 'reverse': #starting waist is horn waist
            w0_horn, z_offset_horn = w0z0(lambd, R_h, w_h) #z_offset is the distance inside the horn (measured from the aperture) where the beam waist falls
            if 'horn_offset' in beam: #we have already started the horn offset list for the different frequencies
                beam['horn_offset'].append(z_offset_horn)
            else: 
                beam['horn_offset'] = [z_offset_horn]
            w0 = [w0_horn]
            z_w0 = [-z_offset_horn]

        
        
        #calculate mirror positions
        for m in range(n_mirrors):
            if m == 0: #first one
                if direction == 'forward':
                    mirror_z = [0]  #forward:  z = 0 at secondary  #could make this 0 at primary instead but would need to edit z_w0 above too
                elif direction == 'reverse':
                    mirror_z = [elelemt_spacings[m] + z_w0[0]]  #reverse: z = 0 at horn aperture
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
        for z in z_list:
            if waist_counter > 0 and z > mirror_z[waist_counter-1]: #we have reached a mirror, move to the next waist
                waist_counter += 1
            z_offset = z-z_w0[waist_counter]
            w_tmp, R_tmp = wR(lambd, z_offset, w0[waist_counter])
            phi_tmp = phi_slippage(lambd, z_offset, w0[waist_counter])
            w.append(w_tmp)
            R.append(R_tmp)
            phi.append(phi_tmp)
        
      
        #store new calculated lists into the per-frequency matrices      
        if y == 0: #we haven't made the per-frequency matrices yet
            w0_matrix = [w0]
            z_w0_matrix = [z_w0]
            w_matrix = [w]
            R_matrix = [R]
            z_list_matrix = [z_list]
            phi_matrix = [phi]
        else:
            w0_matrix.append(w0)
            z_w0_matrix.append(z_w0)
            w_matrix.append(w)
            R_matrix.append(R)
            z_list_matrix.append(z_list)
            phi_matrix.append(phi)
    
    #store everything in the beam dictionary
    beam['w0'] = w0_matrix
    beam['z_w0'] = z_w0_matrix
    beam['w'] = w_matrix
    beam['R'] = R_matrix
    beam['mirror_z'] = mirror_z
    beam['z_list'] = z_list_matrix
    beam['phi'] = phi_matrix
        
    return beam
    
   
    
def thin_lens_plot(beams, direction = 'forward'):
    '''
    
    
    '''
    
    
    plt.ion()
    color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    
    for band in beams:
        beam = beams[band]
        plt.figure()
        for row in range(len(beam['lambda'])):
            label = str(np.round(beam['lambda'][row], 2))
            plt.plot(beam['z_list'][row], beam['w'][row], '.', linewidth = 2, label = label, color = color_list[row])
            
            
            print(beam['mirror_z'])
            #vertical lines and labels for mirrors
            for x in range(len(beam['mirror_z'])):
                plt.vlines(beam['mirror_z'][x], ymin = 0, ymax = beam['mirror_max_radii'][x]/2.5, color = 'black') #5w
                plt.hlines(beam['mirror_max_radii'][x]/2, xmin = beam['mirror_z'][x]-10, xmax = beam['mirror_z'][x]+10, color = 'black') #4w
                if direction == 'forward':
                    plt.text(beam['mirror_z'][x]-20, -10, 'M{}'.format(x+2))
                elif direction == 'reverse':
                    plt.text(beam['mirror_z'][x]-20, -10, 'M{}'.format(len(beam['mirror_z']-x)))
                 
            # vertical lines for beam waists
            for y in range(len(beam['w0'][row])):
                plt.vlines(beam['z_w0'][row][y], ymin = 0, ymax = 10, color = color_list[row], alpha = 0.1, label = label + ' waist')
        
        plt.grid()
        plt.title(band)
        plt.legend()
        plt.tight_layout()
                
        


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
    
    w0 = w/((1+(np.pi*w**2/(lambd*R))**2)**.5)  #goldsmith table 2.3 line 6
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
    w = w_0*np.sqrt(1+(lambd*z/(3.14159*w_0**2))**2)    #goldsmith eq 2.21b and 2.26c
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
    away from the beam waist.  Not currently (5/25) called elsewhere in the code, but
    useful for "by hand" calculations.
    
    Parameters:
    ----------
    lambd: float, wavelength typically in mm
    w: float, 1/e beam width radius at a distance z from the beam waist, typically in mm
    z: float, distance from the beam waist to where w is measured, typically in mm
    
    '''
    
    w0_pos = ((w**2/2) * (1 + ((1 - ((2*lambd*z/(3.1415*w**2))**2))**.5)))**.5
    w0_neg = ((w**2/2) * (1 - ((1 - ((2*lambd*z/(3.1415*w**2))**2))**.5)))**.5
    
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

    zc = 3.14159*(w0_in**2)/lambd
    
    #using the d_out equation (given d_in, d_out, w0_in.  W0_out and f free)
    A = d_out + d_in
    B = (-2*d_in*d_out) - (d_in**2) - (zc**2)
    C = ((d_in**2) * d_out) + (d_out * (zc**2))  # plus or minus sign here is questionable
    
    
    f_pos_d = (-B + ((B**2 - (4*A*C))**.5)) / (2*A)
    f_neg_d = (-B - ((B**2 - (4*A*C))**.5)) / (2*A)
    
    
    #using the w0_out equation.  Given w0_in, w0_out, d_in.  d_out, and f free.  
    A = -1 - ((w0_in/w0_out)**2)
    B = -2*d_in
    C = (zc**2) + (d_in**2)
    
    f_pos_w = (-B + ((B**2 - (4*A*C))**.5)) / (2*A)
    f_neg_w = (-B - ((B**2 - (4*A*C))**.5)) / (2*A)
    
    [w0_out_neg, d_out] = lens(w0_in, d_in, f_neg_w, lambd)
    [w0_out_pos, d_out] = lens(w0_in, d_in, f_pos_w, lambd)
    print('M5 to horn waist dist for waist matching: {}, {}'.format(np.round(w0_out_pos, 2), np.round(w0_out_neg, 2)))
       
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
    
    plt.figure()
    plt.plot(f_list, d_out, label = 'd_out')
    plt.grid()
    plt.xlabel('focal length [mm]')
    plt.tight_layout()
    plt.plot(f_list, w0_out, label = 'w0')
    plt.legend()
    plt.ylabel('w0_out*10 or d_out [mm]')
    plt.tight_layout()
        
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
    
             
    
    

    
    
    

    
    


    
    
    
    
    
def thin_lens_plot_reverse(beams, plot_phi = False):
    '''
    
    
    
    '''
    
    plt.ion()
    plt.figure()
    
    plt.plot(np.array(beams['B6']['z_list']) - beams['B6']['z_elements'][-2], beams['B6']['w'], '-r', linewidth = 2, label = 'B6 beam width')
    plt.plot(np.array(beams['B7']['z_list']) - beams['B7']['z_elements'][-2], beams['B7']['w'], '-b', linewidth = 2, label = 'B7 beam width')
    
    plt.text(-2424, 65, "M3", fontsize = 14, color = 'black')
    plt.text(-3050, 35, "M4", fontsize = 14, color = 'black')
    
    plt.vlines(beams['B6']['z_w0'][2] - beams['B6']['z_elements'][-2], ymin = 0, ymax = 10, color = 'magenta', label = 'B6 waist')
    plt.vlines(beams['B7']['z_w0'][2] - beams['B7']['z_elements'][-2], ymin = 0, ymax = 10, color = 'cyan', label = 'B7 waist')
    
    #beam waists
    plt.vlines(beams['B6']['z_w0'][3] - beams['B6']['z_elements'][-2], ymin = 0, ymax = 10, color = 'magenta')
    plt.vlines(beams['B7']['z_w0'][3] - beams['B7']['z_elements'][-2], ymin = 0, ymax = 10, color = 'cyan')
    
    #mirrors
    plt.vlines(8310, ymin = 0, ymax = 5000, color = 'black') #primary
    plt.hlines(4443, xmin = 8210, xmax = 8410, color='black') # primary 11dB
    plt.vlines(0, ymin=0, ymax = 875, color = 'black') #secondary
    plt.hlines(437.5, xmin = -100, xmax = +100, color='black')
    plt.vlines(beams['B6']['z_elements'][-3] - beams['B6']['z_elements'][-2], ymin = 0, ymax = 35.5, color = 'black') # tertiary
    plt.vlines(beams['B6']['z_elements'][-4] - beams['B6']['z_elements'][-2], ymin = 0, ymax = 20.3, color = 'black') # quaternary
    plt.vlines(beams['B6']['z_elements'][-5] - beams['B6']['z_elements'][-2], ymin = 0, ymax = 7.62, color = 'black') # quinary
    
    
    plt.grid()
    plt.xlabel('distance from secondary [mm]')
    plt.ylabel('beam radius [mm]')
    plt.xlim([-4200,-1800])
    plt.ylim([-10, 150])
    plt.legend()
    
    if plot_phi:
        plt.figure()
        plt.plot(np.array(beams['B6']['z_list']) - beams['B6']['z_elements'][-2], beams['B6']['phi'], '-r', linewidth = 2, label = 'B6 phi')
        plt.plot(np.array(beams['B7']['z_list']) - beams['B7']['z_elements'][-2], beams['B7']['phi'], '-b', linewidth = 2, label = 'B7 phi')
        
        plt.text(-2424, 0, "M3", fontsize = 14, color = 'black')
        plt.text(-3000, 0, "M4", fontsize = 14, color = 'black')
        
        
        plt.xlabel('distance from secondary [mm]')
        plt.ylabel('phi [radians]')
        plt.legend()
    

def thin_lens_plot_forward(beams):
    '''


    '''

    plt.ion()
    plt.figure()
    
  
    #plot beamwidths    
    plt.plot(beams['B6']['z_list'], beams['B6']['w'], '-r', linewidth = 2, label = 'B6 beamwidth')
    plt.plot(beams['B7']['z_list'], beams['B7']['w'], '-b', linewidth = 2, label = 'B7 beamwidth')
    
    #beam waists
    plt.vlines(beams['B6']['z_w0'][2], ymin = 0, ymax = 10, color = 'magenta', label = 'B6 waist')
    plt.vlines(beams['B6']['z_w0'][3], ymin = 0, ymax = 10, color = 'magenta')
    plt.vlines(beams['B7']['z_w0'][2], ymin = 0, ymax = 10, color = 'cyan', label = 'B7 waist')
    plt.vlines(beams['B7']['z_w0'][3], ymin = 0, ymax = 10, color = 'cyan')
    
    #mirrors
    plt.vlines(0, ymin = 0, ymax = 5000, color = 'black') #primary
    plt.hlines(4443, xmin = -20, xmax = 20, color='black') #primary 11dB
    plt.vlines(beams['B6']['z_elements'][1], ymin=0, ymax = 875, color = 'black') #secondary
    plt.vlines(beams['B6']['z_elements'][2], ymin = 0, ymax = 35.5, color = 'black') #tertiary 5w
    plt.hlines(44.375, xmin = beams['B6']['z_elements'][2]-10, xmax = beams['B6']['z_elements'][2]+10, color='black') #tert 4w
    plt.vlines(beams['B6']['z_elements'][3], ymin = 0, ymax = 20.3, color = 'black') #quaternary
    plt.hlines(25.375, xmin = beams['B6']['z_elements'][3]-10, xmax = beams['B6']['z_elements'][3]+10, color='black') #quat 4w
    plt.vlines(beams['B6']['z_elements'][4], ymin = 0, ymax = 7.62, color = 'black') #quinary
    plt.hlines(9.525, xmin = beams['B6']['z_elements'][4]-10, xmax = beams['B6']['z_elements'][4]+10, color='black') #quin 4w
    
    #mirror labels
    plt.text(beams['B6']['z_elements'][2]-20, -10, 'M3', fontsize = 12, color = 'black')
    plt.text(beams['B6']['z_elements'][3]-20, -10, 'M4', fontsize = 12, color = 'black')
    plt.text(beams['B6']['z_elements'][4]-20, -10, 'M5', fontsize = 12, color = 'black')
    plt.text(beams['B6']['z_w0'][-1]-20, -10, 'horn', fontsize = 12, color = 'black')
    
    #tidy plot
    plt.xlim([11450, 13500])
    plt.ylim([-32, 175])
    plt.grid()
    plt.xlabel('distance from primary [mm]')
    plt.ylabel('beam radius [mm]')
    plt.legend()
    plt.tight_layout()

    
    
def reverse_beam_propagation(single_freq_beam, edge_taper_dB = 30, Nz = 100, truncate = 0):
    '''
    
    
    Parameters:
    -----------
    single_freq_beam: dict containing the following keys:
        focal_lengths: list of floats, focal lengths of each optical element [primary, secondary, ...] in mm
        element_spacings: list of floats, distances between each optical element [primary to secondary, secondary to tertiary, ...]
        lambda: float, wavelength in mm
        R_h: float, slant length of horn in mm, equal to the radius of curvature at the horn aperture for conical corrugated horns
        w_h: float, beam radius at the aperture of the horn in mm
        Nz: int, how many points along the z-axis to calculate between each optical element
            e.g. if there are 3 optical elements, then (2*Nz)+1 points will be calculated.  
    edge_taper_dB: int or float, how many dB out to go for the edge tapered radius
    Nz: int, how many points to calculate between elements
    truncate: int or float, z-position (measured from horn) after which to truncate the 
        calculations of w, R, etc. (makes things run faster for optimization stuff, when you don't
        care about the secondary and primary).  If truncate <= 0, then no truncation is done.
    
    
    
    
    Returns:
    --------
    The original dict with the following keys added:
        w0: list of floats, beam waist radii, typically in mm, starting from the horn waist
        z_w0: list of floats, z-positions of each beam waist, typically in mm, starting 
            from the horn waist
        z_elements: list of floats, z-positions of each element, typically in mm, starting 
            from the horn aperture
        z_list: list of floats, every z-position we will calculate w, R, and phi for.  
            0 is at the horn aperture. Typically in mm.  
        w: list of floats, the beam radius at every z-position in z-list.  Typically in mm
        R: list of floats, the wavefront radius of curvatureat every z-position in 
            z-list.  Typically in mm.
        phi: list of floats, the phase shift or "slippage" at every z-position in z-list

    
    '''
    
    focal_lengths = single_freq_beam['focal_lengths']
    focal_lengths.reverse()
    element_spacings = single_freq_beam['element_spacings']
    element_spacings.reverse()
    lambd = single_freq_beam['lambda']
    R_h = single_freq_beam['R_h']
    w_h = single_freq_beam['w_h']
    
    
    #calculate the location of each optical element along the z-axis (optical axis)
    z_elements = [0] #horn aperture is at z=0  #called z_e in Zeng code
    for distance in element_spacings:
        new_z  = z_elements[-1] + distance # add the next distance onto the last z-position
        z_elements.append(new_z)
    

    #calculate the waist radius and z-location of the beam waists between each optical element
    w0_horn, z_offset_horn = w0z0(lambd, R_h, w_h) #z_offset is the distance inside the horn (measured from the aperture) where the beam waist falls
    w0 = [w0_horn]
    z_w0 = [-z_offset_horn] #negative because the waist will be located inside the horn and we're calling the horn aperture z=0  #list of the z-position of each waist
    focal_length_counter = 0
    for element_z in z_elements:
    
        if element_z > 0: # don't do anything with the first one except store it
            d_in = element_z - previous_z_w0 # where is the previous beam waist  #TODO# re-check this for minus signs.  might need to be + instead of -.  
            w0_i, d_out = lens(previous_w0, d_in, focal_lengths[focal_length_counter], lambd)
            w0.append(w0_i)
            z_w0.append(d_out + element_z)
            focal_length_counter += 1            
        previous_z_w0 = z_w0[-1]
        previous_w0 = w0[-1]


    #generate points along z-axis
    for element_z in z_elements:
        if element_z == 0: #for the first one, deal with horn waist location at negative z
            z_list = np.linspace(-z_offset_horn, 0, Nz+1).tolist()
        else:
            if element_z - previous_element_z <= 1000: # for small element spacings, use Nz
                z_steps = np.linspace(previous_element_z, element_z, Nz+1).tolist()[1:] #chop off the first one because it's already in z
            elif element_z - previous_element_z <= 2000: #for medium element spacings, use slightly denser points
                z_steps = np.linspace(previous_element_z, element_z, (Nz*2)+1).tolist()[1:] #chop off the first one because it's already in z
            else: #for big element spacings, use much denser points
                z_steps = np.linspace(previous_element_z, element_z, (Nz*6)+1).tolist()[1:] #chop off the first one because it's already in z
            z_list.extend(z_steps)
        previous_element_z = element_z
    if truncate > 0:
        z_list = [x for x in z_list if x <= truncate]
        
    

    
    #for every point in z, calculate the beam radius (w), radius of curvature (R), phase slippage (phi), and edge taper radius (edge_taper_w)
    w = []
    R = []
    phi = []
    edge_taper_w = []
    
    edge_taper_factor = 0.3393*(edge_taper_dB**.5) #Goldsmith 2.35b
    
    element_number = 1
    for z in z_list:  #step through each z-position in the system
        if z > z_elements[element_number]: #move on to the next element
            element_number += 1
            #print('element number: {}.  z = {}'.format(element_number, z))
        #print(z_w0[element_number], element_number)
        w_tmp, R_tmp = wR(lambd, z-z_w0[element_number-1], w0[element_number-1])   
        phi_tmp = phi_slippage(lambd, z-z_w0[element_number-1], w0[element_number-1])
        w.append(w_tmp)
        edge_taper_w.append(w_tmp*edge_taper_factor)
        R.append(R_tmp)
        phi.append(phi_tmp)
    
    
    single_freq_beam['w0'] = w0
    single_freq_beam['z_w0'] = z_w0
    single_freq_beam['z_elements'] = z_elements
    single_freq_beam['z_list'] = z_list
    single_freq_beam['w'] = w
    single_freq_beam['R'] = R
    single_freq_beam['phi'] = phi
    single_freq_beam['edge_taper_w'] = edge_taper_w
    
    return single_freq_beam
        
    
def forward_beam_propagation(single_freq_beam, Nz = 100, primary_truncate_dB = 11, truncate = 0, calculate_m5 = True):

    focal_lengths = single_freq_beam['focal_lengths']
    element_spacings = single_freq_beam['element_spacings']
    lambd = single_freq_beam['lambda']
    R_h = single_freq_beam['R_h']
    w_h = single_freq_beam['w_h']
    dg = single_freq_beam['greg_foc']


    z_elements = [0] #primary is at z=0
    for distance in element_spacings:
        new_z  = z_elements[-1] + distance # add the next distance onto the last z-position
        z_elements.append(new_z)
        
    #calculate waist radius and z-location of beam waists between each optical element
    z_w0 = [focal_lengths[0], z_elements[1] + dg] #first two waists are at prime and gregorian foci
    w_primary = 5000/(0.3393 * (primary_truncate_dB**.5))  #goldsmith 2.35b, beamwidth at primary
    w0_primary = min(w0_calc(lambd, w_primary, focal_lengths[0])) #primary beam waist radius
    w0_secondary = lens(w0_primary, z_elements[1]-focal_lengths[0], focal_lengths[1], lambd)[0]
    w0 = [w0_primary, w0_secondary]
    focal_length_counter = 2
    for element_z in z_elements:
        if element_z > z_elements[1] and element_z < z_elements[-1]: # don't do anything with the first two or the last one except store it
            d_in = element_z - previous_z_w0 #location of previous beam waist
            w0_out, d_out = lens(previous_w0, d_in, focal_lengths[focal_length_counter], lambd)
            w0.append(w0_out)
            z_w0.append(d_out + element_z)
            focal_length_counter += 1
        previous_z_w0 = z_w0[-1]
        previous_w0 = w0[-1]
    
    print('w0: {}'.format(np.round(w0, 2)))
    print('z_w0: {}'.format(np.round(z_w0, 2)))
     
    #generate points along z-axis
    z_list = [0] #primary position
    previous_element_z = 0
    for element_z in z_elements[1:-1]: #from after the primary up to the last mirror before the horn
        if element_z - previous_element_z <= 500: # for small element spacings, use Nz
            n_points = Nz+1
        elif element_z - previous_element_z <= 2000: #for medium element spacings, use slightly denser points
            n_points = (Nz*2)+1
        else: #for big element spacings, use more points
            n_points = (Nz*6)+1
        z_steps = np.linspace(previous_element_z, element_z, n_points).tolist()[1:] #chop off the first one because it's already in z
        z_list.extend(z_steps)
        previous_element_z = element_z
    
    #add points between horn aperture and horn waist
    z_steps = np.linspace(z_list[-1], z_w0[-1], 200)[1:]
    z_list.extend(z_steps)

    
    if truncate > 0:
        z_list = [x for x in z_list if x <= truncate]
    
    # for every point in z, calculate the beam radius (w), radius of wavefront curvature (R), phase slippage (phi), and edge taper radius (edge_taper_w)
    
    w = []
    R = []
    phi = []
    
    #edge_taper_w = []
    #edge_taper_factor = 0.3393*(edge_taper_dB**.5) #Goldsmith 2.35b
    
    element_number = 1
    for z in z_list: #step through each z-position in the system
        if z > z_elements[element_number] and element_number < len(z_elements)-1: #move on to the next element
            element_number += 1
        w_tmp, R_tmp = wR(lambd, z-z_w0[element_number-1], w0[element_number-1])
        #phi_tmp = phi_slippage(lambd, z-z_w0[element_number-1], w0[element_number-1])
        w.append(w_tmp)
        R.append(R_tmp)
        #phi.append(phi_tmp)
        #edge_taper_w.append(w_tmp*edge_taper_factor)
        
    single_freq_beam['w0'] = w0
    single_freq_beam['z_w0'] = z_w0
    single_freq_beam['z_elements'] = z_elements
    single_freq_beam['z_list'] = z_list
    single_freq_beam['w'] = w
    single_freq_beam['R'] = R
    #single_freq_beam['phi'] = phi
    #single_freq_beam['edge_taper_w'] = edge_taper_w
    
    if calculate_m5:
        
        #calculate horn parameters
        w0_in, z_offset_horn = w0z0(lambd, R_h, w_h) #z_offset is the distance inside the horn (measured from the aperture) where the beam waist falls
        w0_out = w0[3]# waist between M4 and M5
        print('Horn waist for {} mm: {} mm'.format(np.round(lambd,2), np.round(w0_in,2)))
        d_in = z_offset_horn + element_spacings[-1] #distance from horn waist to last mirror
        d_out = z_elements[-2]-z_w0[-2]
        f_M5 = reverse_lens(wo_in, d_in, d_out, w0_out, lambd)  #returns a tuple, both roots
        
        print('Suggested M5 focal lengths for {} mm:  {}'.format(np.round(lambd,2), np.round(f_M5,4)))
        
    return single_freq_beam     


def main_reverse(plot = True):
    '''
    from horn moving towards primary
    
    '''
    
    beams = generate_parameter_dict()
    
    beams['B6'] = reverse_beam_propagation(beams['B6'], edge_taper_dB = 30, Nz = 10000)
    beams['B7'] = reverse_beam_propagation(beams['B7'], edge_taper_dB = 30, Nz = 10000)
    
    M34_waist_diff = (beams['B6']['z_w0'][2] - beams['B6']['z_elements'][-2])-(beams['B7']['z_w0'][2] - beams['B7']['z_elements'][-2])
    M23_waist_diff = (beams['B6']['z_w0'][3] - beams['B6']['z_elements'][-2])-(beams['B7']['z_w0'][3] - beams['B7']['z_elements'][-2])
    
    
    #beam waist offsets between bands
    print('Waist 3-4: {}'.format(np.round(M34_waist_diff, 2)))
    print('Waist 2-3: {}'.format(np.round(M23_waist_diff, 2)))
    
    #beam widths at fixed mirrors
    secondary_index_b6 = beams['B6']['z_list'].index(beams['B6']['z_elements'][4])
    secondary_index_b7 = beams['B7']['z_list'].index(beams['B7']['z_elements'][4])
    w_secondary_b6 = beams['B6']['w'][secondary_index_b6]
    w_secondary_b7 = beams['B7']['w'][secondary_index_b7]
    print('beam radius at secondary: {}, {}'.format( np.round(w_secondary_b6, 2) , np.round(w_secondary_b7, 2) ))
    print('beam radius at primary: {}, {}'.format( np.round(beams['B6']['w'][-1], 2) , np.round(beams['B7']['w'][-1],2)  ))
    
    if plot:
        thin_lens_plot_reverse(beams)
    
    return beams    
    


def constraint(vars, X, z_c, w_oi, w_oo):
    '''
    
    '''
    d_i, d_o, f = vars
    
    eq_spacing = d_i + d_o - X
    
    denominator = ((d_i/f)**2) + ((z_c/f)**2)
    
    eq_d_out = 1 + (((d_i/f)-1) / denominator) - (d_o/f)
    
    eq_w0_out = (w_oi / (denominator**.5)) - w_oo
    
    return [eq_spacing, eq_d_out, eq_w0_out]


def solve_constraint(X, z_c, w_oi, w_oo, initial_guess = (100, 1000, 100)):
    '''
    
    
    '''
    d_i, d_o, f = opt.fsolve(constraint, x0 = initial_guess, args = (X, z_c, w_oi, w_oo)) 
    
    return d_i, d_o, f



    
    
    
    

    
    
