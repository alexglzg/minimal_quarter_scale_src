import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

plt.rcParams['font.size'] = 15 # 20, 36
# plt.rcParams['lines.linewidth'] = 2 # 2, 6
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.figure(figsize=(10,5))

def smooth_data(data_in):
    s = UnivariateSpline(data_in[:,0], data_in[:,1], s=1)
    xs = np.linspace(0, 50000, 1000)
    ys = s(xs)

    data_out = np.zeros((xs.shape[0],2), dtype=np.float32)
    data_out[:,0] = xs
    data_out[:,1] = ys
    return data_out

def compute_time(data_in):
    data_out = np.zeros((0,2), dtype=np.float32)

    sum_time = 0;
    current_node_num = data_in[0,0]
    count = 0

    for i in range(data_in.shape[0]):

        if (current_node_num == data_in[i,0]):
            sum_time = sum_time + data_in[i,1]
            count = count + 1
        else:
            new_line = np.zeros((1,2), dtype=np.float32)
            new_line[0,0] = current_node_num
            new_line[0,1] = sum_time / count;
            data_out = np.append(data_out, new_line, axis=0)

            current_node_num = data_in[i,0]
            sum_time = data_in[i,1]
            count = 1

    new_line = np.zeros((1,2), dtype=np.float32)
    new_line[0,0] = current_node_num
    new_line[0,1] = sum_time / count;
    data_out = np.append(data_out, new_line, axis=0)

    # print data_out
    return smooth_data(data_out)



# Exp 1
# construction_time_1 = compute_time(np.loadtxt('test_1/3_costs/construction_time.txt'))
# plt.plot(construction_time_1[:,0], construction_time_1[:,1], 'k', linewidth=4, label='Construction time')

# test_1_cost_1 = compute_time(np.loadtxt('test_1/1_costs/search_time.txt'))
# test_1_cost_2 = compute_time(np.loadtxt('test_1/2_costs/search_time.txt'))
# test_1_cost_3 = compute_time(np.loadtxt('test_1/3_costs/search_time.txt'))
# plt.plot(test_1_cost_1[:,0], test_1_cost_1[:,1], 'aqua', linewidth=3, label='Search time (one creterion)')
# plt.plot(test_1_cost_2[:,0], test_1_cost_2[:,1]/2, 'coral', linewidth=3, label='Search time (two creteria)')
# plt.plot(test_1_cost_3[:,0], test_1_cost_3[:,1]/2, 'fuchsia', linewidth=3, label='Search time (three creteria)')

# Exp 2
construction_time_2 = compute_time(np.loadtxt('test_2/3_costs/construction_time.txt'))
plt.plot(construction_time_2[:,0], construction_time_2[:,1], 'k', linewidth=4, label='Construction time')

test_2_cost_1 = compute_time(np.loadtxt('test_2/1_costs/search_time.txt'))
test_2_cost_2 = compute_time(np.loadtxt('test_2/2_costs/search_time.txt'))
test_2_cost_3 = compute_time(np.loadtxt('test_2/3_costs/search_time.txt'))
plt.plot(test_2_cost_1[:,0], test_2_cost_1[:,1], 'aqua', linewidth=3, label='Search time (one creterion)')
plt.plot(test_2_cost_2[:,0], test_2_cost_2[:,1]/2, 'coral', linewidth=3, label='Search time (two creteria)')
plt.plot(test_2_cost_3[:,0], test_2_cost_3[:,1]/2, 'fuchsia', linewidth=3, label='Search time (three creteria)')

plt.xlim([0, 50000])
plt.ylim([0, 0.20])
plt.xlabel("Number of nodes")
plt.ylabel("Time (s)")
plt.legend(loc='upper left')

plt.grid(linestyle='--')

# plt.savefig('exp-1.pdf', format='pdf', dpi=400, bbox_inches='tight')
plt.savefig('exp-2.pdf', format='pdf', dpi=400, bbox_inches='tight')

plt.show()