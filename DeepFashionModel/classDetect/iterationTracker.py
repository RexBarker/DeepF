# plot iteration from a given file
# - based on the example from https://pythonprogramming.net/python-matplotlib-live-updating-graphs/

import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

parser = argparse.ArgumentParser()
parser.add_argument('-f',type=str,dest='datafile',required=True,
                    help='(Required) CSV log file to track)')
parser.add_argument('-y1',type=str,dest='y1var',required=False,default='loss',
                    help='(Optional) y1 value to plot)')
parser.add_argument('-y2',type=str,dest='y2var',required=False,default='val_loss',
                    help='(Optional) y2 value to plot)')

args = parser.parse_args()
datafile = args.datafile
y1var = args.y1var
y2var = args.y2var

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


def animate(i):
    with open(datafile,'r') as f:
        pullData = f.read()

    dataArray = pullData.split('\n')
    headers = dataArray.pop(0).split(',')
    dataDict = {k:[] for k in headers}
    #xar = []
    #yar = []
    for eachLine in dataArray:
        if len(eachLine)>1:
            vec= eachLine.split(',')
            for h,v in zip(headers,vec):
                dataDict[h].append(float(v))
            #xar.append(int(x))
            #yar.append(int(y))
    ax1.clear()
    if y2var is None:
        ax1.plot(dataDict['epoch'],dataDict[y1var])
    else:
        ax1.plot(dataDict['epoch'],dataDict[y1var],dataDict['epoch'],dataDict[y2var])

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()