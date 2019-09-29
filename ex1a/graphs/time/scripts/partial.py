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

global_ws = [1,2,4,6,8]



results_tuples = []
x1= [0.030790, 12.131021 , 195.056502]
x2 = [0.017332,6.041834,97.761518]
x4 = [0.011827,3.016668,49.155209]
x6 = [0.010139,2.024827,34.518015]
x8 = [0.009814,1.527707,33.665297]
#results_tuples.append((64,1, x1[0]))
#results_tuples.append((64,2, x2[0]))
#results_tuples.append((64,4, x4[0]))
#results_tuples.append((64,6, x6[0]))
#results_tuples.append((64,8, x8[0]))
results_tuples.append((1024,1, x1[1]))
results_tuples.append((1024,2, x2[1]))
results_tuples.append((1024,4, x4[1]))
results_tuples.append((1024,6, x6[1]))
results_tuples.append((1024,8, x8[1]))
#results_tuples.append((4096,1, x1[2]))
#results_tuples.append((4096,2, x2[2]))
#results_tuples.append((4096,4, x4[2]))
#results_tuples.append((4096,6, x6[2]))
#results_tuples.append((4096,8, x8[2]))
markers = ['.', 'o', 'v', '*', 'D']
fig = plt.figure()
plt.grid(True)
ax = plt.subplot(111)
ax.set_xlabel("$Threads$")
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
	ax.plot(x_ticks, ipc_axis, label="Matrix size "+str(dw)+"x"+str(dw), marker=markers[i%len(markers)])
	i = i + 1

lgd = ax.legend(ncol=len(tuples_by_dw), bbox_to_anchor=(0.9, -0.1), prop={'size':8})
plt.savefig('1024x1024.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
