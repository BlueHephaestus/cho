import itertools#Cartesian product stuff
import numpy as np
from scipy.optimize import minimize

import cho_base
from cho_base import HyperParameter

import cho_config
from cho_config import Configurer

#import sound_notifications
import sms_notifications

"""
BEGIN USER INTERFACE FOR CONFIGURATION
"""
n_y = 20#Number of y values we look at the end of our output
run_count = 3
run_decrement = 0#Amount we decrease the number of runs as we iterate. Will start at initial and never go < 1
epochs = 200
output_types = 1
optimization='momentum'
archive_dir = "../dennis4/data/mfcc_expanded_samples.pkl.gz"#Our input data

output_training_cost = False
output_training_accuracy = False
output_validation_accuracy = True
output_test_accuracy = False
final_test_run = False
#0 = Training Cost, 1 = Training Accuracy, etc.

#Initialize our configurer
configurer = Configurer(epochs, output_types, output_training_cost, output_training_accuracy, output_validation_accuracy, output_test_accuracy, archive_dir)

#Set our initial HPs for cho to search through and optimize
m = cho_base.HyperParameter(10, 10, 0, .1, 1, "Mini Batch Size")
n = cho_base.HyperParameter(0.1, 2.1, 1, .1, 0.01, "Learning Rate")#I really recommend not putting this to 0 by default on accident like I did too many times
u = cho_base.HyperParameter(0.0, 0.0, 0, .1, 0.01, "Optimization Term 1")
v = cho_base.HyperParameter(0.0, 0.0, 0, .1, 0.01, "Optimization Term 2")
l = cho_base.HyperParameter(0.0, 0.0, 0, .1, 0.01, "L2 Regularization Rate")
p = cho_base.HyperParameter(0.0, 0.0, 0, .1, 0.01, "Dropout Regularization Percentage")

hps = [m, n, u, l, p]
"""
END USER INTERFACE FOR CONFIGURATION
"""

n_hp = len(hps)

while True:

    #Get our vectors to make cartesian product out of
    hp_vectors = [hp.get_vector() for hp in hps]#When we need the actual values
    print "New Optimization Initialized, Hyper Parameter Ranges are:"
    #sound_notifications.default_beeps()
    for hp_index, hp in enumerate(hp_vectors):
        print "\t%s: %s" % (hps[hp_index].label, ', '.join(map(str, hp)))

    #Check if we have set them all as constant, once we get our vectors again
    for hp in hp_vectors:
        if len(hp) > 1:
            break
    else:
        print "Optimization Finished, Hyper Parameters are:"
        sms_message = "\nOptimization Finished, Hyper Parameters are:"#\n for header text
        for hp_index, hp in enumerate(hp_vectors):
            print "\t%s: %f" % (hps[hp_index].label, hps[hp_index].min)
            sms_message += "\t%s: %f" % (hps[hp_index].label, hps[hp_index].min)
        sms_message += "\n\t-C.H.O."
        #Send text message notification
        sms_notifications.send_sms(sms_message)

        if final_test_run:
            #Feel free to disable this
            print "Getting Final Optimized Resulting Values..."
            config_avg_result = list(configurer.run_config(run_count, int(hp_vectors[0][0]), hp_vectors[1][0], optimization, hp_vectors[2][0], hp_vectors[3][0], hp_vectors[4][0], hp_vectors[5][0], 419, 69).items())
            print "Final Results with chosen Optimized Values(You should know what output types you enabled so you know what these stand for):"
            for output_type_index, output_type in enumerate(config_avg_result[0][1]):#Just so we know the number
                print "Output Type #%i: %s" % (output_type_index, ', '.join(map(str, [epoch[1][output_type_index] for epoch in config_avg_result[-n_y:]])))

            print "And here are the Optimized Hyper Parameters again:"
            sound_notifications.default_beeps()
            for hp_index, hp in enumerate(hp_vectors):
                print "\t%s: %f" % (hps[hp_index].label, hps[hp_index].min)

        print "Have a nice day."
        break

    #Get cartesian product
    hp_cp = cho_base.cartesian_product(hp_vectors)

    hp_config_count = len(hp_cp)
    hp_cp_results = []#For the results in the cartesian product format, before averaging

    hp_ys = [np.copy(hp_vector).astype(float) for hp_vector in hp_vectors]
    coefs = []#For the coefficients of our quadratic regression of each hyper parameter

    #For our minimization
    #Uses the local min and max of each hp range
    bounds = [(hp.min, hp.max) for hp in hps]
    bounds = tuple(bounds)

    #Since we can just have 1s for each of our hps to plug in here.
    placeholder_hps = [1 for hp in hps]

    #Get our raw cp results/ys
    print "Computing Cartesian Product..."
    for hp_config_index, hp_config in enumerate(hp_cp):

        #Convert our np.float64 types to float
        hp_config = list(hp_config)
        hp_config[1:] = [float(hp) for hp in hp_config[1:]]

        #Execute configuration, get the average entry in the output_dict as a list of it's items
        config_avg_result = list(configurer.run_config(run_count, hp_config[0], hp_config[1], optimization, hp_config[2], hp_config[3], hp_config[4], hp_config[5], hp_config_index, hp_config_count).items())

        #Get our average last n_y values from the respective output_type values
        #Since we shouldn't have any more output types than the one we need, currently
        config_y_vals = [config_y[1][0] for config_y in config_avg_result[-n_y:]]#We have our config_y[1] so we get the value, not the key
        config_avg_y_val = sum(config_y_vals)/float(n_y)

        if not output_training_cost:
            #So we make this into a find-the-minimum one if it's looking at accuracy, which we want to be higher
            config_avg_y_val = 100.0 - config_avg_y_val

        #Add our result to each of our configs in hp_results
        hp_cp_results.append(config_avg_y_val)

    #hp_ys is used to get the average output using our hp caused, so if we had 3 mini batches and 3 regularization rates,
    #our associated hp_y value for our first mini batch size will be the average over the 3 runs that used the first mini batch size.
    #This is where we get those averages.
    print "Averaging respective Hyper Parameter Output..."
    for hp_index, hp in enumerate(hp_vectors):
        for hp_val_index, hp_val in enumerate(hp):
            hp_val_output_sum = 0
            n_hp_val = 0
            for config_index, config in enumerate(hp_cp):
                if hp_val == config[hp_index]:#
                    hp_val_output_sum += hp_cp_results[config_index]
                    n_hp_val += 1
            hp_ys[hp_index][hp_val_index] = hp_val_output_sum/float(n_hp_val)

    #Get our coefficients by doing a quadratic regression on each of our average output for each hyper parameter set
    print "Obtaining Quadratic Regression Coefficients..."
    for hp_index, hp in enumerate(hp_vectors):
        if len(hp) > 1:
            coefs.append(np.polynomial.polynomial.polyfit(hp, hp_ys[hp_index], 2))
        else:
            coefs.append([hp[0], 0, 0])

    print "Coefficients are %s" % (', '.join(map(str, coefs)))

    print "Computing Minimum of Multivariable Hyper Parameter Function..."

    #Our main function to minimize once we have our coefficients
    hp_function = lambda hps: sum([coef[0] + coef[1]*hp + coef[2]*hp**2 for coef, hp in zip(coefs, hps)])#Why the fuck are quadratic regression coefficient orders backwards

    res = minimize(hp_function, placeholder_hps, bounds=bounds, method='TNC', tol=1e-10, options={'xtol': 1e-8, 'disp': False})

    #Now our res.x are our new center point values
    center_points = res.x

    print "Minimum values are: %s" % (', '.join(map(str, center_points)))
    #print center_points

    print "Computing new Hyper Parameter Optimization Ranges..."
    for hp_index, center_point in enumerate(center_points):
        if len(hp_vectors[hp_index]) > 1:
            step = hps[hp_index].step
            step_decrease_factor = hps[hp_index].step_decrease_factor
            stop_threshold = hps[hp_index].stop_threshold
            new_step = step*step_decrease_factor

            #print new_step, stop_threshold
            if new_step < stop_threshold:
                #Time to mark this value as final and stop modifying.
                #This means we no longer update it, we just replace the min and max with our center point,
                #and make the step 0. Just as we do with dependent variables at the start
                new_min = center_point
                new_max = center_point
                new_step = 0
            else:
                #We get our inclusive range, i.e if center point is 19.14, 
                #We'd get 14.14, and 24.14. Then we round up and round down respectively,
                #to get 15 and 24
                new_min = cho_base.step_roundup(center_point-(step*.5), new_step)
                new_max = cho_base.step_rounddown(center_point+(step*.5), new_step)

            #We update with our new params if an independent hyper parameter
            hps[hp_index].min = new_min
            hps[hp_index].max = new_max
            hps[hp_index].step = new_step
            #print new_min, new_max

    #Decrement our run number
    if run_count > 1: 
        run_count -= run_decrement

    
