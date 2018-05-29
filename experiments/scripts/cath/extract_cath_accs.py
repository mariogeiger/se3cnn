import sys
import glob
import os
import re
import numpy as np
import argparse

logfiles = sys.argv[1:]

acc_regexp = re.compile('acc=([0-9]*\.[0-9]*)')
loss_regexp = re.compile('loss=([0-9]*\.[0-9]*)')
reduction_regexp = re.compile('x([0-9]+)')
convdropout_regexp = re.compile('convdropout_([0-9.]+)')
l1l2_regexp = re.compile('L1L2_1e-([0-9]+)')
model_regexp = re.compile('model\s+([a-zA-Z0-9_]+)')
dataset_regexp = re.compile('data_filename\s+([a-zA-Z0-9_/.-]+)')
convdropout_regexp = re.compile('p_drop_conv\s+([a-zA-Z0-9_/.-]+)')
L1_regexp = re.compile('lamb_conv_weight_L1\s+([a-zA-Z0-9_/.-]+)')

parser = argparse.ArgumentParser()

parser.add_argument("--uncertainty", default=0.02, type=float,
                    help="For each run, test-accuracy averages will be calculated over all models which have a validation score within this uncertainty from the maximum accuracy on the validation set. (default: %(default)s)")
parser.add_argument("--min-values", default=100, type=int,
                    help="Log files will only be considered if they have at least this number of observations (in order to skip incomplete runs). (default: %(default)s)")
parser.add_argument("--burnin", default=0, type=int,
                    help="Number of initial values to discard. (default: %(default)s)")

args, logfiles = parser.parse_known_args()


def extract_max_acc_by_uncertainty(acc_data, burnin, uncertainty=0.02):
    '''Find all values that lie within a certain uncertainty from the best value'''

    indices = list(range(len(acc_data)))
    idx_value_list = zip(acc_data[burnin:], indices[burnin:])

    best_values, best_indices = zip(*sorted(idx_value_list, reverse=True))

    end_index = list(np.array(best_values) < (best_values[0]-uncertainty)).index(True)

    best_values = best_values[:end_index]
    best_indices = best_indices[:end_index]
                                    
    return best_values, best_indices    


values = {}
for logfile in logfiles:
    
    output_filename = logfile
    
    with open(output_filename) as f:
        validation_content = [line for line in f.readlines() if "VALIDATION SET" in line]
    
    loss_data = [float(loss_regexp.search(line).group(1)) for line in validation_content]
    acc_data = [float(acc_regexp.search(line).group(1)) for line in validation_content]
    
    with open(output_filename) as f:
        test_content = [line for line in f.readlines() if "TEST SET" in line]
    
    test_loss_data = [float(loss_regexp.search(line).group(1)) for line in test_content]
    test_acc_data = [float(acc_regexp.search(line).group(1)) for line in test_content]
    
    model_type = ""
    with open(output_filename) as f:
        match = model_regexp.search(f.read())
        if match:
            model_type = match.group(1)
    
    data_filename = ""
    with open(output_filename) as f:
        match = dataset_regexp.search(f.read())
        if match:
            data_filename = os.path.basename(match.group(1))
    
    convdropout = ""
    with open(output_filename) as f:
        match = convdropout_regexp.search(f.read())
        if match:
            convdropout = match.group(1)

    L1 = ""
    with open(output_filename) as f:
        match = L1_regexp.search(f.read())
        if match:
            L1 = match.group(1)
    
    reduction_factor = 1
    match = reduction_regexp.search(data_filename)
    if match:
        reduction_factor = float(match.group(1))

    if model_type not in values:
        values[model_type] = []

    if len(acc_data) < args.min_values:
        print("Skipping", logfile)
        continue

    best_accs, best_accs_indices = extract_max_acc_by_uncertainty(acc_data, burnin=args.burnin, uncertainty=args.uncertainty)

    best_test_accs = [test_acc_data[idx] for idx in best_accs_indices]

    values[model_type].append((reduction_factor, np.average(best_test_accs)))

    print("model={:s} reduction_factor={:d} logfile={:s} len={:d} convdropout={:s} L1L2_reg={:s} val_acc_best_mean={:.3} test_acc={:.3}".format(model_type, int(reduction_factor), logfile, len(acc_data), convdropout, L1, np.average(best_accs), np.average(best_test_accs)))
    
    

for model_type in values:
    data = np.array(values[model_type])

    import pandas as pd
    d = pd.DataFrame(data)
    mean = d.groupby(0).mean()
    mean.columns = ['mean']
    std = d.groupby(0).std()
    std.columns = ['std']

    overview = pd.concat([mean, std], axis=1).reset_index()
    overview.rename(columns={0:'reduction factor'}, inplace=True)

    print(model_type)
    print(overview)
    

