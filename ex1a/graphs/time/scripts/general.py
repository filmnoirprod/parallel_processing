#!/usr/bin/env python

import sys, os
import itertools, operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def tuples_by_dispatch_width(tuples):
	ret = []
	tuples_sorted = sorted(tuples, key=operator.itemgetter(0))
	for key,group in itertools.groupby(tuples_sorted,operator.itemgetter(0)):
		ret.append((key, zip(*map(lambda x: x[1:], list(group)))))
	return ret

global_ws = [64,1024,4096]



results_tuples = []
x1= [0.030790, 12.131021 , 195.056502]
x2 = [0.017332,6.041834,97.761518]
x4 = [0.011827,3.016668,49.155209]
x6 = [0.010139,2.024827,34.518015]
x8 = [0.009814,1.527707,33.665297]
results_tuples.append((1, 64, x1[0]))
results_tuples.append((2, 64, x2[0]))
results_tuples.append((4, 64, x4[0]))
results_tuples.append((6, 64, x6[0]))
results_tuples.append((8, 64, x8[0]))
results_tuples.append((1, 1024, x1[1]))
results_tuples.append((2, 1024, x2[1]))
results_tuples.append((4, 1024, x4[1]))
results_tuples.append((6, 1024, x6[1]))
results_tuples.append((8, 1024, x8[1]))
results_tuples.append((1, 4096, x1[2]))
results_tuples.append((2, 4096, x2[2]))
results_tuples.append((4, 4096, x4[2]))
results_tuples.append((6, 4096, x6[2]))
results_tuples.append((8, 4096, x8[2]))

markers = ['.', 'o', 'v', '*', 'D']
fig = plt.figure()
plt.grid(True)
ax = plt.subplot(111)
ax.set_xlabel("$N$")
ax.set_ylabel("$Time (s)$")

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
	ax.plot(x_ticks, ipc_axis, label="#threads_"+str(dw), marker=markers[i%len(markers)])
	i = i + 1

lgd = ax.legend(ncol=len(tuples_by_dw), bbox_to_anchor=(0.9, -0.1), prop={'size':8})
plt.savefig('katikalo.ipc.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
