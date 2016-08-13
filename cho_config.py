#So we can import without being there
import sys
sys.path.append("../dennis/dennis4/src")

#For Cho's use optimizing
import dennis4
from dennis4 import *

import json
import sample_loader
from sample_loader import *

class Configurer(object):
    def __init__(self, epochs, output_types, output_training_cost, output_training_accuracy, output_validation_accuracy, output_test_accuracy, archive_dir):
        self.epochs = epochs

        self.training_data, self.validation_data, self.test_data = get_data_subsets(p_training = 0.8, p_validation = 0.1, p_test = 0.1, archive_dir=archive_dir)
        self.training_data, self.validation_data, self.test_data, self.normalize_data = load_data_shared(training_data=self.training_data, validation_data=self.validation_data, test_data=self.test_data, normalize_x=True)

        #Our default values
        self.input_dims = 51*51
        self.output_dims = 6
        self.output_types = output_types#DON'T FORGET TO UPDATE THIS WITH THE OTHERS
        self.forced_scheduler_interval = 69
        self.scheduler_check_interval = 50
        self.param_decrease_rate = 2
        self.training_data_subsections=None#Won't be needing this for our tiny dataset!

        self.early_stopping=False
        self.automatic_scheduling=True

        self.output_training_cost=output_training_cost
        self.output_training_accuracy=output_training_accuracy
        self.output_validation_accuracy=output_validation_accuracy
        self.output_test_accuracy=output_test_accuracy

        self.output_title="Cho Tests"
        self.output_filename="cho_tests"
        self.print_results = False
        self.print_perc_complete = False
        self.update_output = True
        self.save_net = False

    def run_config(self, run_count, mini_batch_size, learning_rate, optimization, optimization_term1, optimization_term2, regularization_rate, p_dropout, global_config_index, config_index, config_count ):#Last two are for progress
        mini_batch_size = int(mini_batch_size)
        if p_dropout < 0:
            p_dropout = 0

        #We run one config each time from cho, adding to our output dict each time

        #Gotta make seperate Network instances for each run else the params don't get re-initialized
        '''
        Configs I don't want to have to recreate again when I need them

                    [
                        [Network([ 
                            FullyConnectedLayer(n_in=self.input_dims, n_out=300, activation_fn=sigmoid, p_dropout=p_dropout), 
                            FullyConnectedLayer(n_in=300, n_out=80, activation_fn=sigmoid, p_dropout=p_dropout), 
                            FullyConnectedLayer(n_in=80, n_out=20, activation_fn=sigmoid, p_dropout=p_dropout), 
                            FullyConnectedLayer(n_in=20, n_out=self.output_dims, activation_fn=linear, p_dropout=p_dropout)], mini_batch_size, cost=quadratic), mini_batch_size, 
                            learning_rate, optimization, optimization_term1, optimization_term2, regularization_rate, self.scheduler_check_interval, self.param_decrease_rate, ""] 
                        for r in range(run_count)
                    ],
        '''
        configs = [
                    [
                        [Network([ 
                            ConvPoolLayer(image_shape=(mini_batch_size, 1, 51, 51),
                                filter_shape=(64, 1, 4, 4),
                                poolsize=(2,2),
                                poolmode='average_exc_pad'),
                            ConvPoolLayer(image_shape=(mini_batch_size, 64, 24, 24),
                                filter_shape=(128, 64, 5, 5),
                                poolsize=(2,2),
                                poolmode='average_exc_pad'),
                            FullyConnectedLayer(n_in=10*10*128, n_out=600, activation_fn=sigmoid, p_dropout=p_dropout), 
                            FullyConnectedLayer(n_in=600, n_out=30, activation_fn=sigmoid, p_dropout=p_dropout), 
                            SoftmaxLayer(n_in=30, n_out=self.output_dims, p_dropout=p_dropout)], mini_batch_size, cost=log_likelihood), mini_batch_size, 
                            learning_rate, optimization, optimization_term1, optimization_term2, regularization_rate, self.forced_scheduler_interval, self.scheduler_check_interval, self.param_decrease_rate, ""] 
                        for r in range(run_count)
                    ],
                 ]

        #First, we run our configuration
        output_dict = {}
        #for config_index, config in enumerate(configs):
        for run_index in range(run_count): 
            output_dict[run_index] = {}
            net = configs[global_config_index][run_index][0]
            net.output_config(
                output_filename=self.output_filename, 
                training_data_subsections=self.training_data_subsections, 
                early_stopping=self.early_stopping,
                automatic_scheduling=self.automatic_scheduling,
                output_training_cost=self.output_training_cost,
                output_training_accuracy=self.output_training_accuracy,
                output_validation_accuracy=self.output_validation_accuracy,
                output_test_accuracy=self.output_test_accuracy,
                print_results=self.print_results,
                print_perc_complete=self.print_perc_complete,
                config_index=config_index,
                config_count=config_count,
                run_index=run_index,
                run_count=run_count,
                output_types=self.output_types)

            output_dict = net.SGD(output_dict, self.training_data, self.epochs, mini_batch_size, learning_rate, self.validation_data, self.test_data, optimization=optimization, optimization_term1=optimization_term1, optimization_term2=optimization_term2, lmbda=regularization_rate, forced_scheduler_interval=self.forced_scheduler_interval, scheduler_check_interval=self.scheduler_check_interval, param_decrease_rate=self.param_decrease_rate)
        

        #Save layers if we choose to
        if self.save_net:
            print "Saving Neural Network Layers..."
            net.save('../saved_networks/%s.pkl.gz' % self.output_filename)
            f = open('../saved_networks/%s_metadata.txt' % (self.output_filename), 'w')
            f.write("{0}\n{1}\n{2}".format(self.normalize_data[0], self.normalize_data[1], self.input_dims))
            f.close()

        #After all runs have executed
        #If there were more than one runs for this configuration, we average them all together for a new one
        #We do this by looping through all our y values for each epoch, and doing our usual mean calculations
        if run_count > 1:
            output_dict[run_count+1] = {}#For our new average entry
            for j in range(self.epochs):
                output_dict[run_count+1][j] = []#For our new average entry
                for o in range(self.output_types):
                    avg = sum([output_dict[r][j][o] for r in range(run_count)]) / run_count
                    output_dict[run_count+1][j].append(avg)
            return output_dict[run_count+1]#Return our average end result
        else:
            return output_dict[0]#Return our run, since we only did one.
