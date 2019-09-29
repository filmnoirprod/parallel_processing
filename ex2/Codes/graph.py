#!/usr/bin/env python

import sys, os
import itertools, operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def printer(computation,communication,title,name,end,step):

	plt.clf()
	plt.cla()
	N = 3
	ind = np.arange(N)    # the x locations for the groups
	width = 0.65       # the width of the bars: can also be len(x) sequence

	p1 = plt.bar(ind,computation, width,color='b')
	p2 = plt.bar(ind,communication,width,bottom=computation,color='red')

	plt.ylabel('Time (sec)')
	plt.title(title)
	plt.xticks(ind, ('Jacobi','GaussSeidelSor','RedBlackSor'))
	plt.yticks(np.arange(0,end,step))
	plt.legend((p1[0], p2[0]), ('Computation', 'Communication'))
	plt.savefig(name, bbox_inches='tight')
	return

def reader(myfile):

	total1 = np.array([])
	total2 = np.array([])
	total3 = np.array([])
	comp1 = np.array([])
	comp2 = np.array([])
	comp3 = np.array([])
	print(myfile)
	with open(myfile) as f:
		lines = f.readlines()
		for line in lines:
			parts = line.split(" ")
			if(len(parts)>1):
				if(int(parts[2])==2048):
					total1=np.append(total1,float(parts[14]))
					comp1=np.append(comp1,float(parts[12]))
				elif(int(parts[2])==4096):
					total2=np.append(total2,float(parts[14]))
					comp2=np.append(comp2,float(parts[12]))
				elif(int(parts[2])==6144):
					total3=np.append(total3,float(parts[14]))
					comp3=np.append(comp3,float(parts[12]))
		f.close()
		return total1,total2,total3,comp1,comp2,comp3

def caller(file1,file2,file3):

	t1,t2,t3,c1,c2,c3 = reader(file1)

	t12,t22,t32,c12,c22,c32 = reader(file2)

	t1 = t1+t12
	t2 = t2+t22
	t3 = t3+t32
	c1 = c1+c12
	c2 = c2+c22
	c3 = c3+c32

	t12,t22,t32,c12,c22,c32 = reader(file3)

	t1 = t1+t12
	t2 = t2+t22
	t3 = t3+t32
	c1 = c1+c12
	c2 = c2+c22
	c3 = c3+c32

	myInt = 3
	t1 = t1/myInt
	t2 = t2/myInt
	t3 = t3/myInt
	c1 = c1/myInt
	c2 = c2/myInt
	c3 = c3/myInt

	communication1 = t1-c1
	communication2 = t2-c2
	communication3 = t3-c3

	return t1,t2,t3,c1,c2,c3,communication1,communication2,communication3

def printbarplots():

	# array size 6144
	printer((cj3[6],cg3[6],cr3[6]),(talkj3[6],talkg3[6],talkr3[6]),'N=6144 P=64','6144_64.png',40,5)
	printer((cj3[5],cg3[5],cr3[5]),(talkj3[5],talkg3[5],talkr3[5]),'N=6144 P=32','6144_32.png',40,5)
	printer((cj3[4],cg3[4],cr3[4]),(talkj3[4],talkg3[4],talkr3[4]),'N=6144 P=16','6144_16.png',40,5)
	printer((cj3[3],cg3[3],cr3[3]),(talkj3[3],talkg3[3],talkr3[3]),'N=6144 P=8','6144_8.png',40,5)
	#printer((cj3[2],cg3[2],cr3[2]),(talkj3[2],talkg3[2],talkr3[2]),'N=6144 P=4','6144_4.png')
	#printer((cj3[1],cg3[1],cr3[1]),(talkj3[1],talkg3[1],talkr3[1]),'N=6144 P=2','6144_2.png')
	#printer((cj3[0],cg3[0],cr3[0]),(talkj3[0],talkg3[0],talkr3[0]),'N=6144 P=1','6144_1.png')

	# array size 4096
	printer((cj2[6],cg2[6],cr2[6]),(talkj2[6],talkg2[6],talkr2[6]),'N=4096 P=64','4096_64.png',40,5)
	printer((cj2[5],cg2[5],cr2[5]),(talkj2[5],talkg2[5],talkr2[5]),'N=4096 P=32','4096_32.png',40,5)
	printer((cj2[4],cg2[4],cr2[4]),(talkj2[4],talkg2[4],talkr2[4]),'N=4096 P=16','4096_16.png',40,5)
	printer((cj2[3],cg2[3],cr2[3]),(talkj2[3],talkg2[3],talkr2[3]),'N=4096 P=8','4096_8.png',40,5)
	#printer((cj2[2],cg2[2],cr2[2]),(talkj2[2],talkg2[2],talkr2[2]),'N=4096 P=4','4096_4.png')
	#printer((cj2[1],cg2[1],cr2[1]),(talkj2[1],talkg2[1],talkr2[1]),'N=4096 P=2','4096_2.png')
	#printer((cj2[0],cg2[0],cr2[0]),(talkj2[0],talkg2[0],talkr2[0]),'N=4096 P=1','4096_1.png')

	# array size 2048
	printer((cj1[6],cg1[6],cr1[6]),(talkj1[6],talkg1[6],talkr1[6]),'N=2048 P=64','2048_64.png',5,0.2)
	printer((cj1[5],cg1[5],cr1[5]),(talkj1[5],talkg1[5],talkr1[5]),'N=2048 P=32','2048_32.png',5,0.2)
	printer((cj1[4],cg1[4],cr1[4]),(talkj1[4],talkg1[4],talkr1[4]),'N=2048 P=16','2048_16.png',5,0.2)
	printer((cj1[3],cg1[3],cr1[3]),(talkj1[3],talkg1[3],talkr1[3]),'N=2048 P=8','2048_8.png',20,2)
	#printer((cj1[2],cg1[2],cr1[2]),(talkj1[2],talkg1[2],talkr1[2]),'N=2048 P=4','2048_4.png')
	#printer((cj1[1],cg1[1],cr1[1]),(talkj1[1],talkg1[1],talkr1[1]),'N=2048 P=2','2048_2.png')
	#printer((cj1[0],cg1[0],cr1[0]),(talkj1[0],talkg1[0],talkr1[0]),'N=2048 P=1','2048_1.png')
	print("Bar plotting completed succesfully !")

def tuples_by_dispatch_width(tuples):
	ret = []
	tuples_sorted = sorted(tuples, key=operator.itemgetter(0))
	for key,group in itertools.groupby(tuples_sorted,operator.itemgetter(0)):
		ret.append((key, zip(*map(lambda x: x[1:], list(group)))))
	return ret

def printgraphs(results_tuples,filename,title):

	global_ws = [1,2,4,8,16,32,64]

	plt.clf()
	plt.cla()
	markers = ['.', 'o', 'v', '*', 'D']
	fig = plt.figure()
	plt.grid(True)
	plt.title(title)
	ax = plt.subplot(111)
	ax.set_xlabel("$Cores$")
	ax.set_ylabel("$Speedup$")

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
		ax.plot(x_ticks, ipc_axis, label="Array_size "+str(dw)+"x"+str(dw), marker=markers[i%len(markers)])
		i = i + 1

	lgd = ax.legend(ncol=len(tuples_by_dw), bbox_to_anchor=(1.0, -0.1), prop={'size':8})
	plt.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')


def lastplotter(x,x1,x2):
	results_tuples = []
	results_tuples.append((2048,1,x[0]/x[0]))
	results_tuples.append((2048,2,x[0]/x[1]))
	results_tuples.append((2048,4,x[0]/x[2]))
	results_tuples.append((2048,8,x[0]/x[3]))
	results_tuples.append((2048,16,x[0]/x[4]))
	results_tuples.append((2048,32,x[0]/x[5]))
	results_tuples.append((2048,64,x[0]/x[6]))
	results_tuples.append((4096,1,x1[0]/x1[0]))
	results_tuples.append((4096,2,x1[0]/x1[1]))
	results_tuples.append((4096,4,x1[0]/x1[2]))
	results_tuples.append((4096,8,x1[0]/x1[3]))
	results_tuples.append((4096,16,x1[0]/x1[4]))
	results_tuples.append((4096,32,x1[0]/x1[5]))
	results_tuples.append((4096,64,x1[0]/x1[6]))
	results_tuples.append((6144,1,x2[0]/x2[0]))
	results_tuples.append((6144,2,x2[0]/x2[1]))
	results_tuples.append((6144,4,x2[0]/x2[2]))
	results_tuples.append((6144,8,x2[0]/x2[3]))
	results_tuples.append((6144,16,x2[0]/x2[4]))
	results_tuples.append((6144,32,x2[0]/x2[5]))
	results_tuples.append((6144,64,x2[0]/x2[6]))
	return results_tuples


j1,j2,j3,cj1,cj2,cj3,talkj1,talkj2,talkj3=caller('jacobi1','jacobi2','jacobi3')
g1,g2,g3,cg1,cg2,cg3,talkg1,talkg2,talkg3=caller('gauss1','gauss2','gauss3')
r1,r2,r3,cr1,cr2,cr3,talkr1,talkr2,talkr3=caller('red1','red2','red3')
print("Done reading files,now let's plot em!")

# uncomment in order to print bar plots
#printbarplots()

# uncomment in order to print line plots
#res1 = lastplotter(j1,j2,j3)
#res2 = lastplotter(g1,g2,g3)
#res3 = lastplotter(r1,r2,r3)
#printgraphs(res1,'jacobi_speedup.png','Jacobi MPI speedup')
#printgraphs(res2,'gauss_speedup.png','GaussSeidelSor MPI speedup')
#printgraphs(res3,'redblack_speedup.png','RedBlackSor MPI speedup')






