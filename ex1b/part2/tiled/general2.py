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
x1= [0.5302,3.5819,28.5976]
x2 = [0.2903,1.8372,14.5832]
x4 = [0.1578,0.9960,7.6048]
x8 = [0.912,0.5553,3.9334]
x16 = [0.0753,0.3456,2.0598]
x32 = [0.0812,0.2491,1.7751]
x64 = [0.0860,0.2613,1.0413]

results_tuples.append((1, 1024, x1[0]))
results_tuples.append((2, 1024, x2[0]))
results_tuples.append((4, 1024, x4[0]))
results_tuples.append((8, 1024, x8[0]))
results_tuples.append((16, 1024, x16[0]))
results_tuples.append((32, 1024, x32[0]))
results_tuples.append((64, 1024, x64[0]))

results_tuples.append((1, 2048, x1[1]))
results_tuples.append((2, 2048, x2[1]))
results_tuples.append((4, 2048, x4[1]))
results_tuples.append((8, 2048, x8[1]))
results_tuples.append((16, 2048, x16[1]))
results_tuples.append((32, 2048, x32[1]))
results_tuples.append((64, 2048, x64[1]))

results_tuples.append((1, 4096, x1[2]))
results_tuples.append((2, 4096, x2[2]))
results_tuples.append((4, 4096, x4[2]))
results_tuples.append((8, 4096, x8[2]))
results_tuples.append((16, 4096, x16[2]))
results_tuples.append((32, 4096, x32[2]))
results_tuples.append((64, 4096, x64[2]))



markers = ['.', 'o', 'v', '*', 'D']
fig = plt.figure()
plt.grid(True)
ax = plt.subplot(111)
ax.set_xlabel("$ Threads $")
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
plt.savefig('tiled2.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
