#####################################################################################
#    Filename : readme.txt
#    Date : Aug 26, 2015
#    What : The readme file for Proj_Resonance_Tangent_GitHub_Run_Me.ipynb
#           and def_resonance_tangent_for_github.py
#          
#
#####################################################################################

What else is here :
================================

"Proj_Resonance_Tangent_GitHub_Run_Me.ipynb" 
+
"def_resonance_tangent_for_github.py"

=

Together, they will read a file containing a list of numbers, and they will tell if that list of numbers have any resonances in it. Additionally, it will then fit those minimas to a tangent curve, and it will tell you what the chi-squared of all those individual minimas are. 

“2015-07-17-resonance-search.npz” : the file I tested on

“All_the_plots.zip” : all the tangent and resonance plots of the test


What you need to put into this :
================================
	 
A) A file, and you would have to specify where it is. As an example, I have left in the  programme where I would have used to locate the file on my computer. You can just cut and paste to get to where the file is on your computer. An example file ".npz" is attached here.
   
B) You need to tell the programme what the frequency and data number are. 
This is usually shortened to 'freq' and 'data' respectively, without the parenthesis.
   
C) Finally, you will get 3 options on how you may want to look at the Lorentzian
.1) This is just all the resonances close up
.2) This is just all the resonances close up and they would be fitted with the Lorentzian (pick me!)
.3) This is just all the resonances close up, followed by all the resonances close up that are fitted with the Lorentzian 
       

A word of caution:
===================
I wouldn't change anything in this programme except these 8 parameters when looking for resonance:
''' Just toggle these 8 parameters'''
'''You just have to toggle these 8 values : These 8 parameters define the search for minimums'''
flat = 3                       # flat is the gradient of the points around centre
move = 0                       # move is the number of points around the centre
noise = 3                      # noise is how many points away from 'move'
tol = 1                        # tolerance determine spread of tangent func to left
tol2 = 1                       # tolerance determine spread of tangent func to right
grad_threshold_4_res = 8       # less than this absolute amount, and it is just noise
noise_depth = 0.17             # this is the depth of "line" ie how much noise 
random_spread = 6*noise        # how far along hori_axis you wanna check:random noise


How to take this code forward:
===================
For the version of the main code “Proj_Resonance_Tangent_GitHub_Run_Me” dated on Aug28 2015, the best way that I can think of to take this code forward would be to look at the definitions page i.e. “def_resonance_tangent_for_github.py”. 

1) Line 497 : I would try to update “def tangent(x,p)”, which is the main tangent function which we are fitting to. In particular, I would try to change co_eff, which I think is not that well-defined.

2) Line 508 : I would correspondingly look at “def fit_me_to_tangent(champ,x,y,move,noise,tol,tol2,flat,count_chi,count_chi_less_than_1)”. In particular, I would look at the initial values p[0], p[1], p[2] and p[3]. The main problem I think is p[1] = gamma. I eventually used p[1] to define co_eff, but there is no real reason too, except that it seems to work . Moreover, gamma is a kind of Full Width Half Max, but again, there’s no real reason to have that in a tangent. 

3) Line 532 : I would also try to improve on “y_ideal = -1*tangent(x_ideal,p)”. This is the idealized negative tangent which you are trying to fit to. Alas, you sometimes have to fit to an idealized positive tangent, if that is what you get in the measured experiment. Maybe you can write a simple if loop using “champ” from above, whereby if y[champ+1] is a local maxima, it means that the next point is bigger than the minima, and therefor you need a negative tangent ; correspondingly, if y[champ-1] is a local maxima, it means that the previous points is bigger than the minima, and therefor, you need a positive tangent. 


