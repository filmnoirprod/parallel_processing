#!/usr/bin/env python

import sys, os
import itertools, operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def tuples_by_dispatch_width(tuples):
	ret = []
	tuples_sorted = sorted(tuples, key=operator.itemgetter(1))
	for key,group in itertools.groupby(tuples_sorted,operator.itemgetter(1)):
		ret.append((key, zip(*map(lambda x: x[1:], list(group)))))
	return ret

global_ws = [1,2,4,8,16,32,64]



results_tuples = []
x1= [0.5587,3.9195,29.2988]
x2 = [0.3735,2.2700,15.7942]
x4 = [0.3538,1.8976,12.4867]
x8 = [0.3584,2.0103,11.9880]
x16 = [0.3581,1.9432,12.3261]
x32 = [0.3679,2.0033,11.7582]
x64 = [0.3712,2.0139,12.5791]

results_tuples.append((1, 1024, x1[0]/x1[0]))
results_tuples.append((2, 1024, x1[0]/x2[0]))
results_tuples.append((4, 1024, x1[0]/x4[0]))
results_tuples.append((8, 1024, x1[0]/x8[0]))
results_tuples.append((16, 1024, x1[0]/x16[0]))
results_tuples.append((32, 1024, x1[0]/x32[0]))
results_tuples.append((64, 1024, x1[0]/x64[0]))

results_tuples.append((1, 2048, x1[1]/x1[1]))
results_tuples.append((2, 2048, x1[1]/x2[1]))
results_tuples.append((4, 2048, x1[1]/x4[1]))
results_tuples.append((8, 2048, x1[1]/x8[1]))
results_tuples.append((16, 2048, x1[1]/x16[1]))
results_tuples.append((32, 2048, x1[1]/x32[1]))
results_tuples.append((64, 2048, x1[1]/x64[1]))

results_tuples.append((1, 4096, x1[2]/x1[2]))
results_tuples.append((2, 4096, x1[2]/x2[2]))
results_tuples.append((4, 4096, x1[2]/x4[2]))
results_tuples.append((8, 4096, x1[2]/x8[2]))
results_tuples.append((16, 4096, x1[2]/x16[2]))
results_tuples.append((32, 4096, x1[2]/x32[2]))
results_tuples.append((64, 4096, x1[2]/x64[2]))



markers = ['.', 'o', 'v', '*', 'D']
fig = plt.figure()
plt.grid(True)
ax = plt.subplot(111)
ax.set_xlabel("$ Threads $")
ax.set_ylabel("$ Speedup $")

i = 0
tuples_by_dw = tuples_by_dispatch_width(results_tuples)
for tuple in tuples_by_dw:
	dw = tuple[0]
	ws_axis = tuple[1][0]
	ipc_axis = tuple[1][1]
	x_ticks = np.arange(0, len(global_ws))
	x_labels = map(str, global_ws)
	ax.xaxis.set_ticks(x_ticks)
	ax.xaxis.set_ticklabels(x_labels)
	#ax.yaxis.set_ticks(np.arange(0,210,10))

	print x_ticks
	print ipc_axis
	ax.plot(x_ticks, ipc_axis, label="Matrix size "+str(dw)+"x"+str(dw), marker=markers[i%len(markers)])
	i = i + 1

lgd = ax.legend(ncol=len(tuples_by_dw), bbox_to_anchor=(0.9, -0.1), prop={'size':8})
plt.savefig('recursivespeedup.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
