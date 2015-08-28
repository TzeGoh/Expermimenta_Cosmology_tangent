#####################################################################################
#    Filename : def_resonance_tangent_for_github.py
#    Date : Aug 28, 2015
#    What : Main list of definitions for the main file 
#           "Proj_Resonance_Lorentzian_GitHub_Run_Me.ipynb"
#####################################################################################

'''Definitions by AstroTze'''
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import mode
from scipy.optimize import leastsq


'''This will give you a list of seperation of data points that could be minimas. You have to give it the data'''
def give_me_possible_seperation_of_minima(data):
    data = 20*np.log10(np.abs(data))
    depth_list = []
    for i in range(1,len(data)):
        depth = abs( data[i] - data[i - 1] )
        depth_list.append(depth)
    ave_depth = np.mean(depth_list)
    min_list = []
    #print ave_depth,"ave_depth"
    for i in range(1,len(data)):
        if  abs( data[i] - data[i - 1] ) > 5*ave_depth:
            min_list.append(i)
            #print i
    for i in range(1,len(min_list)):
        if min_list[i] - min_list[i-1]==1:
            min_list[i] = 0
    #print min_list
    while 0 in min_list : min_list.remove(0)
    return min_list

'''You give this a list of suspected minimas, and it returns for you a list that removes periodicity
and even gives you a list of starting number and its repition to remove'''
def give_me_seperation_and_repitition(min_list):
    seperation_list=[]
    sep_and_repeat_list = []
    
    for i in range(1,len(min_list)):
        diff = min_list[i] - min_list[i-1]
        seperation_list.append(diff)
        
    #print seperation_list
   
    repeated_seperation = mode(seperation_list)
    while repeated_seperation[1][0] > 1:
        sep_and_repeat_list.append([repeated_seperation[0][0],repeated_seperation[1][0]])
        while repeated_seperation[0][0] in seperation_list :
            seperation_list.remove(repeated_seperation[0][0])
        repeated_seperation = mode(seperation_list)

    #print sep_and_repeat_list
    
    if sep_and_repeat_list != []:
        for i in range(len(sep_and_repeat_list)):
            #print sep_and_repeat_list[i][1]
            if sep_and_repeat_list[i][1] / len(seperation_list) < 0.1 : # it means less than this percent
                sep_and_repeat_list[i] = [0,0]
    while [0, 0] in sep_and_repeat_list : sep_and_repeat_list.remove([0,0])
    return sep_and_repeat_list


'''This will give you the list of starting number and the seperation which you can use to cancel out periodicity'''
def give_me_starting_number_and_seperation(min_list,sep_and_repeat_list):
    cancelling_list=[]
    #print min_list
    #print sep_and_repeat_list
    for i in range(len(sep_and_repeat_list)):
        for j in range(1,len(min_list)):
            if min_list[j] - min_list[j-1] == sep_and_repeat_list[i][0]:
                starting_number = min_list[j-1]
                break
            else:
                starting_number = 0
        if starting_number != 0:
            cancelling_list.append([starting_number,sep_and_repeat_list[i][0]])      
    return cancelling_list


'''This gives you two maximas (one to left, one to right) away from minima(Champ)'''
def give_me_2_maximas(champ,y,move,noise,tol):
    away = move + noise
    temp_champ = champ                           
    hl = 0                             
    hr = 0                             
    while y[temp_champ ] <= y[temp_champ - tol*away] and temp_champ !=0:            
        temp_champ = temp_champ - 1                        
        hl = hl +1                   
    temp_champ = champ                          
    while y[temp_champ] <= y[temp_champ + tol*away] and temp_champ != len(y)-tol*away:            
        temp_champ = temp_champ + 1                        
        hr = hr +1
    return hl,hr

'''Gives you two points above 2 Full Width Half Max, counting from Minima to Maxima'''
def give_me_2_half_maxes(champ,y,h_fwhm): 
    j = champ                    
    xl = 0                       
    xr = 0                       
    while y[j] < h_fwhm:         
        j = j-1                  
        xl = xl + 1               
    j = champ                    
    while y[j] < h_fwhm:         
        j = j+1                  
        xr = xr + 1               
    return xl,xr

'''This gives you the all important gamma'''
def give_me_gamma(x,champ,xl,xr):
    gam1 = abs(abs(x[champ - xl]) - abs(x[champ]))
    gam2 = abs(abs(x[champ + xr]) - abs(x[champ]))
    gamma=(gam1 + gam2)/2
    return gamma


'''This code returns will return the average gradient to the right side of the curve'''   
def right_gradient_average(x,y,center,move,away):
    dx = np. gradient(x[center + move : center + away])
    right_gradient = np.gradient(y[center + move: center + away],dx)
    right_gradient_ave = np.average(right_gradient)
    return right_gradient_ave 

'''This code returns will return the average gradient to the left side of the curve''' 
def left_gradient_average(x,y,center,move,away):
    dx = np. gradient(x[center - away +1 : center - move +1])
    left_gradient = np.gradient(y[center - away : center - move],dx)
    left_gradient_ave = np.average(left_gradient)
    return left_gradient_ave
    
'''This defines the very local minimum up to only the noise'''
def is_it_local_minimum(y,center,noise):
    y_list = y[center-noise : center + noise]
    if y[center] == min(y_list):
        return "yes"    
    
'''This returns ALL the local minimums'''
def minima(x, y, move, noise, flat):                # move points before and after minimum
    locmins = []
    away = move + noise
    for center in range(away, len(y) - (away)):    # noiseth point after move is checked
        
        right_gradient_ave = right_gradient_average(x,y,center,move,away)
        left_gradient_ave = left_gradient_average(x,y,center,move,away)
        if abs(left_gradient_ave) > flat or abs(right_gradient_ave) > flat:
            if abs(left_gradient_ave) < 100 or abs(right_gradient_ave) < 100:
                if is_it_local_minimum(y,center,noise) == "yes":
                    locmins.append([ x[center], y[center] ])
       
    return locmins  



'''This removes any local minimums that are the same and next to each other '''
def remove_repeated(locmins):
    for i in range(1,len(locmins)):
        if locmins[i][1] == locmins[i-1][1]:        # The second index [1] is the y-axis!
            locmins[i][0]=0
            locmins[i][1]=0 
    while [0,0] in locmins: locmins.remove([0,0])
    return locmins

'''This code ranks the local minimums for you, according to indexes in y '''
def rank_the_locmins(locmins,y):
    for i in range(len(y)):                         # Sorts locmins into their position
        for j in range (len(locmins)):
            if y[i] == locmins[j][1]:
                if len(locmins[j])==2:              # Prevents picking up repeated points
                    locmins[j].append(i)
    return locmins

'''This will sort out the locmins and remove any minimas which are too close'''
def remove_close_by_mins(locmins, move, noise):
    for i in range(1,len( locmins)):
        after = locmins[i][2]
        before = locmins[i-1][2]
        diff = abs(after - before)
        away = move + noise
        if diff < away :
            y_after = locmins[i]
            y_before = locmins[i-1]
            if y_after < y_before :
                locmins[i-1] = [0,0,0]
            if y_after > y_before :
                locmins[i] = [0,0,0]
    while [0,0,0] in locmins: locmins.remove([0,0,0])
    return locmins

'''This removes any points that have gradient less steep that the grad_threshold_4_res'''
def remove_not_resonance(locmins, x, y, noise, grad_threshold_4_res):
    all_the_grads_above_threshold=[]
    maxima = 0
    for i in range(len(locmins)):
        rank = locmins[i][2]
        y_in_locmins = locmins[i][1]
        for j in range(len(y)):
            if y_in_locmins == y[j]:
                grad_of_locmin_right = ( y[j+1] - y[j] ) / (x[j+1] - x[j])
                #print grad_of_locmin_right,"grad_of_locmin_right"
                grad_of_locmin_left = ( y[j] - y[j-1] ) / (x[j] - x[j-1])
                #print grad_of_locmin_left,"grad_of_locmin_left"
                if abs(grad_of_locmin_left) > grad_threshold_4_res or abs(grad_of_locmin_right) > grad_threshold_4_res:
                    all_the_grads_above_threshold.append([grad_of_locmin_left,grad_of_locmin_right])
                else:
                    locmins[i]=[0, 0, 0]
                #print rank
    #print "all_the_grads_above_threshold : left, right"
    #for i in range(len(all_the_grads_above_threshold)):
    #    print all_the_grads_above_threshold[i]
    while [0,0,0] in locmins: locmins.remove([0,0,0])
    return locmins


'''This removes any noise according the the thickness of the line ie the noise '''
def remove_just_noise(locmins, y, noise, noise_depth):
    just_noise=[]
    maxima = 0
    for i in range(len(locmins)):
        rank = locmins[i][2]
        y_in_locmins = locmins[i][1]
        for j in range(len(y)):
            if y_in_locmins == y[j]:
                y_list = y[j - noise : j + noise]
                #print y[j+1], y[j]
                if abs(max(y_list) - min(y_list)) > noise_depth:
                    just_noise.append(abs(max(y_list) - min(y_list)))
                else:
                    locmins[i]=[0, 0, 0]
                #print rank
    #print "all_the_just_noise : max minus min"
    #for i in range(len(just_noise)):
    #    print just_noise[i]
    while [0,0,0] in locmins: locmins.remove([0,0,0])
    return locmins


'''This removes any random fluctuations to a spread of an interger times the noise '''
def remove_random_minimum(locmins, y, noise, random_spread):
    non_random_list=[]
    for i in range(len(locmins)):
        rank = locmins[i][2]
        y_in_locmins = locmins[i][1]
        for j in range(len(y)):
            if y_in_locmins == y[j]:
                y_list=[]
                for k in range(random_spread):
                    y_list = y[j-random_spread : j+random_spread]
                    if y[j] == min(y_list):
                        non_random_list.append(y[j])
                    else:
                        locmins[i]=[0, 0, 0]
    while [0,0,0] in locmins: locmins.remove([0,0,0])              
    #print "the non_random_list is :"
    #for i in range(len(non_random_list)):
    #    print non_random_list[i]
    return locmins

'''This removes any points that is 100 times steeper than the median gradient '''
def remove_too_high_gradient(locmins, x, y):
    all_the_grads_list = []
    maxima = 0
    for i in range(len(locmins)):
        rank = locmins[i][2]
        y_in_locmins = locmins[i][1]
        for j in range(len(y)):
            if y_in_locmins == y[j]:
                grad_of_locmin_right = ( y[j+1] - y[j] ) / (x[j+1] - x[j])
                #print grad_of_locmin_right,"grad_of_locmin_right"
                grad_of_locmin_left = ( y[j] - y[j-1] ) / (x[j] - x[j-1])
                #print grad_of_locmin_left,"grad_of_locmin_left"
                ave_gradient = ( abs(grad_of_locmin_right) + abs(grad_of_locmin_left) ) / 2
                all_the_grads_list.append(ave_gradient)
    all_the_grad_ave = np.median(all_the_grads_list)
    for i in range(len(all_the_grads_list)):
        if all_the_grads_list[i] > 100 * all_the_grad_ave:
            locmins[i] = [0, 0, 0]
    #print all_the_grads_list
    #print all_the_grad_ave
    while [0,0,0] in locmins: locmins.remove([0,0,0])
    return locmins

'''Let us find the range of frequencies to look at : if the freq are too closed, the range is the same '''
def ranges(locmins):
    range_of_freq = []
    for i in range(len(locmins)):
        range_of_freq.append(locmins[i][2])
    for i in range(1,len(range_of_freq)):
        if range_of_freq[i] - range_of_freq[i-1] < 50:          ###arbitrary value
            range_of_freq[i-1] = 0
    while 0 in range_of_freq : range_of_freq.remove(0)
    return range_of_freq 



'''This code just allows me to look at ranges of values for the lorentzian '''
def ranges_to_look(ranges):
    list_of_differences = []
    for i in range(1, len(ranges)):
        difference = ranges[i] - ranges[i - 1]
        list_of_differences.append(difference)
    #print list_of_differences,'list_of_differences'
    ave_diff = int(np.mean(list_of_differences))
    #print ave_diff  
    list_of_ranges = []
    for i in range(len(ranges)):
        if i == 0:
            begin = ranges[i] - ave_diff / 6
            end = ranges[i] + (ranges[i + 1]- ranges[i]) / 6
            this_range = [begin, end]
            list_of_ranges.append(this_range)
        if i == len(ranges) - 1:
            begin = ranges[i] - (ranges[i] - ranges[i - 1]) / 6
            end = ranges[i] + ave_diff / 6
            this_range = [begin, end]
            list_of_ranges.append(this_range)
        elif i != 0 and i != len(ranges) -1 :
            begin =  ranges[i] - ( ranges[i] - ranges[i - 1] )/6
            end = ranges[i] + (ranges[i + 1]- ranges[i]) / 6
            this_range = [begin, end]
            list_of_ranges.append(this_range)
    #print list_of_ranges
    return list_of_ranges

'''This code should tell you all the resonators, according to the Lorentzian definition'''
def resonators(flat, move, noise, tol, x, y, grad_threshold_4_res, noise_depth, random_spread):
    order_of_mins = []
    locmins = minima(x, y, move, noise,flat)   
    remove_repeated(locmins)                              
    locmins = rank_the_locmins(locmins,y) 
    
    locmins = remove_close_by_mins(locmins,move,noise)
    
    locmins = remove_not_resonance(locmins, x, y, noise, grad_threshold_4_res)
    locmins = remove_just_noise(locmins, y, noise, noise_depth)
    locmins = remove_random_minimum(locmins, y, noise, random_spread)
    locmins = remove_too_high_gradient(locmins, x, y)
    
    for i in range(len(locmins)):
        order_of_mins.append([i+1, locmins[i][0], locmins [i][1]])
    return locmins, order_of_mins

'''This code will convert any rankings into frequency'''
def convert_rank_to_freq(ranges_to_look, x):
    freq_to_look= []
    for i in range(len(ranges_to_look)):
        j_begin = ranges_to_look[i][0]
        j_end = ranges_to_look[i][1]
        freq_begin = x[j_begin]
        freq_end = x[j_end]           
        freq_index = [freq_begin, freq_end]
        freq_to_look.append(freq_index)
    return freq_to_look

'''This should the range of values to the min and max of the data'''
def convert_rank_to_min_and_max_data(ranges_to_look,y):
    data_to_look = []
    for i in range(len(ranges_to_look)):
        begin = ranges_to_look[i][0]
        end = ranges_to_look[i][1]
        data_max = max(y[begin:end])
        data_min = min(y[begin:end])
        data_interval = (data_max - data_min)/10
        data_max = data_max + data_interval
        data_min = data_min - data_interval
        data_to_look.append ([data_min,data_max])
    return data_to_look

'''This should give you all the limiting ranges to look at'''
def give_me_all_limits(ranges_to_look,freq_to_look,data_to_look,j):
    ranges_begin,ranges_end = ranges_to_look[j][0], ranges_to_look[j][1]
    xbegin, xend = freq_to_look[j][0] ,freq_to_look[j][1]
    ybegin, yend = data_to_look[j][0], data_to_look[j][1]
    return ranges_begin, ranges_end, xbegin, xend, ybegin, yend

'''This lets you count the number of data points in your range'''
def count_no_of_data_points_per_close_range(order_of_mins,x,ranges_begin,ranges_end,freq_points,index,index_list,i):
    if order_of_mins[i][1] in x[ranges_begin: ranges_end]:
        freq_points.append(order_of_mins[i][1])
        index = index + 1
        index_list.append(index)
    return freq_points, index, index_list

'''This code should give you 2 decimal places'''
def give_me_two_decimal(freq_points):
    freq_points_dec = []
    for j in range(len(freq_points)):
        freq_to_2_dec = "{0:.2f}".format(freq_points[j])
        freq_points_dec.append(freq_to_2_dec)  
    return freq_points_dec

'''This tells you where to look for the resonances'''
def where_do_i_look(locmins,x,y):
    ranging = ranges(locmins)
    where_to_look = ranges_to_look(ranging)
    freq_to_look = convert_rank_to_freq(where_to_look,x)
    data_to_look = convert_rank_to_min_and_max_data(where_to_look,y)
    return where_to_look, freq_to_look, data_to_look


'''This tells you initially where to look, and how to box the overall plot in'''
def box_in_the_plot(ranges_to_look, freq_to_look, data_to_look):
    ystart = 0
    yend = 0
    for j in range(len(ranges_to_look)):
        xstart,xend = freq_to_look[0][0],freq_to_look[j][1]
        ystart = ystart + data_to_look[j][0]
        yend = yend + data_to_look[j][1]
    ystart = ystart / len(data_to_look)
    yend = yend / len(data_to_look)

    x_interval =abs( xend - xstart )/ len(freq_to_look)
    y_interval =abs( yend - ystart )

    xstart = xstart - 2*x_interval
    xend = xend + 2*x_interval
    ystart = ystart - y_interval
    yend = yend + y_interval
    return xstart, xend, ystart, yend

'''This shows you the main plot'''
def main_plot_with_noise(x,y,xstart,xend,ystart,yend):
    plt.plot(x,y)
    plt.title('Main plot with noise removed')
    plt.xlim(xstart,xend )
    plt.ylim(ystart,yend)  
    
'''This shows you where the resonators are'''
def show_me_resonators(order_of_mins,data,x,y,xstart,xend,ystart,yend):
    print 'length of data',len(data)
    print "There are",len(order_of_mins),"resonators, and they occur at ...'"
    print "Resonator","\t", "Frequency", "\t", "data numbers"
    for i in range(len(order_of_mins)):
        print order_of_mins[i][0],'\t','\t', order_of_mins[i][1],'\t', order_of_mins[i][2]
    plt.plot(x, y)
    plt.title ( 'Plot with the Resonators')   
    plt.xlim(xstart,xend )
    plt.ylim(ystart,yend)                            
    ax = plt.gca()                                              
    for i in range(len(order_of_mins)):
        ax.axvline(order_of_mins[i][1], color = 'red',linewidth = 2, alpha = 0.7)         
        plt.scatter (order_of_mins[i][1], order_of_mins[i][2], color = 'red', s = 40)

        
        
'''This should show you all the resonators in close range'''                   
def show_me_resonators_in_close_range(ranges_to_look,freq_to_look,data_to_look,x,y,order_of_mins):
    index = 0
    index_list=[]
    for j in range(len(ranges_to_look)):
        ranges_begin, ranges_end, xbegin, xend, ybegin, yend \
        = give_me_all_limits(ranges_to_look, freq_to_look, data_to_look,j)
        plt.plot(x,y)
        plt.scatter(x,y, alpha = 0.4)
        plt.xlim (xbegin, xend)
        plt.ylim (ybegin, yend)
        freq_points = []
        ax = plt.gca()
        
        for i in range(len(order_of_mins)):
            ax.axvline(order_of_mins[i][1], color = 'red')         
            plt.scatter (order_of_mins[i][1], order_of_mins[i][2], color = 'red')  
            freq_points, index, index_list = count_no_of_data_points_per_close_range\
            (order_of_mins,x,ranges_begin,ranges_end,freq_points,index,index_list,i)
            
        freq_points_dec = give_me_two_decimal(freq_points)    
        title = " The data point is", index_list,"...and the frequency is", freq_points_dec
        plt.title(title, fontsize = 15)
        plt.xlabel('Frequency') 
        plt.ylabel('Data')
        index_list = []
        plt.figure()        

'''This returns the Lorentzian function'''
def neg_Lorentz(x, p):                            
    Numerator = p[2]                             #p[0] = Considered as 'x0': centre of x  
    Denominator = ((x - p[0])**2 + p[2])         #p[1] = (gamma*np.pi) 
    Co_eff = 1/p[1]                              #p[2] = gamma**2 
    Background = p[3]                            #p[3] =  Considered as 'y0': background
    return (-1 * Co_eff * (Numerator/Denominator)) + Background   
    
    
    
    
    

'''Returns the difference between an ideal, and a measured y.Used for chi-square'''   
def residuals(p,y_meas, x_ideal):  
    err = y_meas - tangent(x_ideal,p)        # y_meas - y_ideal
    return err 

        
'''This returns the tangent function'''
def tangent(x, p):                                
    b = 1/p[0]                                    # x0 = center of x
    c = p[2]                                      #p[2] = c =phase shift
    x0 = b*x + c                                  #p[0] = where pi/2 spreads out to
    tang = np.tan(x0)                               
    co_eff = 2*p[1]                               #p[1] = (gamma*np.pi) = FWHM
    background = p[3]                             #p[3] =  Considered as 'y0': background
    return ( 0.1 *co_eff* tang - background) 
        
'''This fits for you the lorentzian, and will tell you which points have chi-squared more 
than 1, and which ones have more than 100 '''
def fit_me_to_tangent(champ,x,y,move,noise,tol,tol2,flat,count_chi,count_chi_less_than_1):
            
    hl,hr =  give_me_2_maximas(champ,y,move,noise,tol)# hl, hr : height to left, right  
    h_ave = int((hl + hr )/2)
    hl = h_ave + tol*noise
    hr = h_ave + tol2*noise 
    gamma = ( abs(y[champ] - y[champ-hl]) + abs(y[champ +1 ] - y[champ + 1 + hr]) )/ 4 
                                                      # kinda full_width_half_max  
    x_val_right = x[champ + hl]
    x_val_left = x[champ - hl]
    x_sep_right = x[champ + hl] - x[champ]
    x_sep_left = x[champ] - x[champ - hl]
    x_sep_ave = (x_sep_right + x_sep_left) / 2
    #print hl, hr, x_sep_left, x_sep_right, x_sep_ave, " hl, hr, x_sep_left, x_sep_right, x_sep_ave"
    
    
    p=[0.0, 0.0, 0.0, 0.0]                            
    p[0] = (2/np.pi) * x_sep_ave                                             
    p[1] = gamma
    p[2] = np.pi/2
    p[3] = y[champ + hr]
        #print j+1,p, "The original parameters"  
    
    x_ideal = np.linspace(x[champ - hl], x[champ + hr], hl+hr)# x_fit = x_ideal = x_meas
    y_ideal = -1*tangent(x_ideal,p)
    y_meas = y[champ - hl : champ + hr]
    x_meas = x_ideal                                      
    
    from scipy.optimize import leastsq
    plsq = leastsq(residuals, p, args=(y_meas,x_meas))      
        #print j+1,(plsq[0]), "The parameters for leastsq"   
    x_fit = x_ideal 
    y_fit = tangent(x_ideal,plsq[0])
        
    chi_squared=0
    for item in range(len(y_meas)):
        element = (y_meas[item]-y_fit[item])**2
        chi_squared = chi_squared + element
    if chi_squared > 100:
        count_chi = count_chi +1
    if chi_squared < 1:
        count_chi_less_than_1=count_chi_less_than_1 +1
            
    return chi_squared, count_chi, count_chi_less_than_1, hl, hr, champ, x_ideal, y_ideal, x_fit, y_fit
        
'''This should show you all the resonators in close range fitted to the lorentzian function'''
def show_me_resonators_in_close_range_with_tangent(ranges_to_look,freq_to_look,data_to_look,x,y,\
                                                      move,noise,tol,tol2,flat,order_of_mins,locmins):
    index = 0
    index_list=[]
    count_chi=0
    count_chi_less_than_1=0
    chi_squared_list =[]
    chi_squared_total = []
    freq_points_total = []
    for j in range(len(ranges_to_look)):
        ranges_begin, ranges_end, xbegin, xend, ybegin, yend \
        = give_me_all_limits(ranges_to_look, freq_to_look, data_to_look,j) 
        plt.plot(x,y)
        plt.scatter(x,y, alpha = 0.4, label = "Measured")
        plt.xlim (xbegin, xend)                                                    ## Put it normal
        plt.ylim (ybegin, yend)                                                    # at the end
        freq_points = []
        ax = plt.gca()
        for i in range(len(order_of_mins)):
            ax.axvline(order_of_mins[i][1], color = 'red')         
            plt.scatter(order_of_mins[i][1], order_of_mins[i][2], color = 'red')  
            freq_points, index, index_list = count_no_of_data_points_per_close_range\
            (order_of_mins,x,ranges_begin,ranges_end,freq_points,index,index_list,i)        
            
        '''This plots the tangent per minima'''
        for i in range(len(freq_points)):    
            freq_points_total.append(freq_points[i])
        locmins_list=[]
        chi_squared_list =[]
        for i in range(len(index_list)):
            locmins_list.append(locmins[index_list[i]-1])   
        for j in range(len(locmins_list)):
            champ = locmins_list[j][2]
            chi_squared, count_chi, count_chi_less_than_1, hl, hr, champ, x_ideal, y_ideal, x_fit, y_fit \
            = fit_me_to_tangent(champ,x,y,move,noise,tol,tol2,flat,count_chi,count_chi_less_than_1) 
            
            chi_squared_list.append(chi_squared)
            chi_squared_total.append(chi_squared)
            
            plt.scatter(x[champ - hl], y[champ - hl], color = 'orange', s=180 , alpha = 0.8 )
            plt.scatter(x[champ + hr], y[champ + hr], color = 'orange', s=180 , alpha = 0.8)
            plt.plot(x_ideal, y_ideal, color = 'orange', linewidth = 4, linestyle = '--',alpha = 0.8)  
            plt.scatter(x[champ - hl], y[champ - hl], color = 'orange', s=180 , alpha = 0.8 )
            plt.scatter(x[champ + hr], y[champ + hr], color = 'orange', s=180 , alpha = 0.8)
            plt.plot(x_ideal, y_ideal, color = 'orange', linewidth = 4, linestyle = '--',alpha = 0.8) 
            plt.plot(x_fit, y_fit, color = 'green', linewidth = 4 , alpha = 0.3 )
            plt.scatter(x_fit, y_fit, color = 'green', alpha = 0.3 ) 
        #print chi_squared_list
               
#########################################################################################################              
        '''Everything below is just for labelling !!!'''
        if len(locmins_list)!=0:
            plt.scatter(x[champ - hl], y[champ - hl], color = 'orange', s=180 , alpha = 0.8, label = "Maxima" )    
            plt.scatter (locmins[j][0], locmins[j][1], color = 'red', label = "Minima")    
            plt.plot(x_ideal, y_ideal, color = 'orange', linewidth = 4, linestyle = '--', alpha = 0.8 , label = " Ideal" )
            plt.plot(x_fit, y_fit, color = 'green', linewidth = 4 , alpha = 0.3 , label = "Fit" )
            plt.legend(loc = 4)
            
        freq_points_dec = give_me_two_decimal(freq_points) 
        chi_squared_dec = give_me_two_decimal(chi_squared_list) 
        title = " Resonator is", index_list,"...and Chi-square is", chi_squared_dec,"...frequency is",freq_points_dec
        plt.title(title, fontsize = 15)
        plt.xlabel('Frequency') 
        plt.ylabel('Data')
        index_list = []
        plt.figure()
        
    return count_chi, count_chi_less_than_1, chi_squared_total,freq_points_total, len(chi_squared_total) 

''' This will tell you all the chi-squared that appears'''
def print_me_those_chi_square(count_chi,count_chi_less_than_1,chi_squared_total,freq_points_total):
    print "The number of fits with Chi_square more than 100 is", count_chi 
    print "The number of fits with Chi_square less than 1 is", count_chi_less_than_1 
    print "Resonator",'\t','Chi_square','\t','at these Frequencies'
    for i in range(len(chi_squared_total)):
        print i+1, '\t','\t',chi_squared_total[i],'\t', freq_points_total[i]
        

    
