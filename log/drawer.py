import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import seaborn as sns
import numpy as np

f = open('log_with_baseline','r+')
s = f.read()

line = re.split('Iteration ',s)
avg_r = []
for i in xrange(len(line)-1):
    kk = line[i+1]
    tmp = re.split('=', kk)
    spl = re.split('\n', tmp[1])
    avg_r.append(float(spl[0]))

f = open('log_without_baseline','r+')
s = f.read()

line = re.split('Iteration ',s)
avg_r_2 = []
for i in xrange(len(line)-1):
    kk = line[i+1]
    tmp = re.split('=', kk)
    spl = re.split('\n', tmp[1])
    avg_r_2.append(float(spl[0]))

plt.plot(avg_r, label='with baseline')
plt.hold
plt.plot(avg_r_2, label='without baseline')
plt.xlabel('iteration')
plt.ylabel('avg return')
plt.show()

