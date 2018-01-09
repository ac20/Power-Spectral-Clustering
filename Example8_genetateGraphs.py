import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

try:
    colnames = ['scale', 'Process_time','Rcut_time','accRcut','AMI_Rcut','ARI_Rcut','PRcut_time','accPRcut','AMI_PRcut','ARI_PRcut']
    data_8a = pd.read_csv("results_8a.csv", header = None )
    data_8b = pd.read_csv("results_8b.csv", header = None )
    data_8c = pd.read_csv("results_8c.csv", header = None )

    data_8a.columns = colnames
    data_8b.columns = colnames
    data_8c.columns = colnames
except:
    colnames = ['scale', 'Rcut_time','accRcut','AMI_Rcut','ARI_Rcut','PRcut_time','accPRcut','AMI_PRcut','ARI_PRcut']
    data_8a = pd.read_csv("results_8a.csv", header = None )
    data_8b = pd.read_csv("results_8b.csv", header = None )
    data_8c = pd.read_csv("results_8c.csv", header = None )

    data_8a.columns = colnames
    data_8b.columns = colnames
    data_8c.columns = colnames


scale = np.array(list(data_8a['scale']) + list(data_8b['scale']) + list(data_8c['scale']))

nPixels  = [int(x*610)*int(x*340) for x in list(data_8a['scale'])]
nPixels +=  [int(x*1096)*int(x*715) for x in list(data_8b['scale'])]
nPixels +=  [int(x*512)*int(x*217) for x in list(data_8c['scale'])]
nPixels = np.array(nPixels)

Rcut_time = np.array(list(data_8a['Rcut_time']) + list(data_8b['Rcut_time']) + list(data_8c['Rcut_time']))
PRcut_time = np.array(list(data_8a['PRcut_time']) + list(data_8b['PRcut_time']) + list(data_8c['PRcut_time']))

accRcut = np.array(list(data_8a['accRcut']) + list(data_8b['accRcut']) + list(data_8c['accRcut']))
accPRcut = np.array(list(data_8a['accPRcut']) + list(data_8b['accPRcut']) + list(data_8c['accPRcut']))

ymin, ymax = np.min(PRcut_time), np.max(PRcut_time)
xmin, xmax = np.min(Rcut_time), np.max(Rcut_time)

Tmin = min(np.min(PRcut_time), np.min(Rcut_time))
Tmax = max(np.max(PRcut_time), np.max(Rcut_time))

plt.figure()
plt.axis([Tmin, Tmax, Tmin, Tmax])
plt.scatter(Rcut_time[np.where(scale==0.7)], PRcut_time[np.where(scale==0.7)], color='b', marker='d', label = 'nPixels = x0.7')
plt.scatter(Rcut_time[np.where(scale==0.8)], PRcut_time[np.where(scale==0.8)], color='b', marker='d', label = 'nPixels = x0.8')
plt.scatter(Rcut_time[np.where(scale==0.9)], PRcut_time[np.where(scale==0.9)], color='b', marker='d', label = 'nPixels = x0.9')
plt.plot(np.arange(Tmin,Tmax,0.001),np.arange(Tmin,Tmax,0.001), color='r')
plt.xlabel('Ratio cut (time in sec)')
plt.ylabel('Power Ratio cut  (time in sec)')
# plt.legend()
plt.savefig('./img/Example8_plot.pdf')

plt.close('all')


Tmin = min(np.min(accPRcut), np.min(accRcut))
Tmax = max(np.max(accPRcut), np.max(accRcut))
plt.figure()
plt.axis([Tmin, Tmax, Tmin, Tmax])
plt.scatter(accRcut[np.where(scale==0.7)], accPRcut[np.where(scale==0.7)], color='b', marker='d', label = 'x0.7')
plt.scatter(accRcut[np.where(scale==0.8)], accRcut[np.where(scale==0.8)], color='c', marker='x', label = 'x0.8')
plt.scatter(accRcut[np.where(scale==0.9)], accRcut[np.where(scale==0.9)], color='g', marker='o', label = 'x0.9')
plt.plot(np.arange(Tmin,Tmax,0.001),np.arange(Tmin,Tmax,0.001), color='r')
plt.xlabel('Ratio cut (accuracy)')
plt.ylabel('Power Ratio cut (accuracy)')
plt.legend()
plt.savefig('./img/Example8_plotAccuracy.pdf')

plt.close('all')

from sklearn.linear_model import LinearRegression as LR

clf = LR(fit_intercept=True)
a =  np.log(nPixels)
a = a.reshape((a.shape[0],1))

plt.figure()
plt.scatter(nPixels/100000., Rcut_time, color='b', label='Ratio cut')
clf.fit(a, np.log(Rcut_time))
b = np.log(np.sort(nPixels)).reshape(-1,1)
plt.plot(np.sort(nPixels)/100000.,np.exp(clf.predict(b)),'b')

plt.scatter(nPixels/100000., PRcut_time, color='g', label='Power Ratio cut')
clf.fit(a, np.log(PRcut_time))
b = np.log(np.sort(nPixels)).reshape(-1,1)
plt.plot(np.sort(nPixels)/100000.,np.exp(clf.predict(b)),'g')

plt.xlabel('Number of pixels in 10^5')
plt.ylabel('Time  (in sec)')
plt.legend()
plt.savefig('./img/Example8_plotb.pdf')

plt.close('all')

d = [data_8a, data_8b, data_8c]
for scale in [0.7,0.8,0.9]:
    for l in range(3) :
        print '********************************'
        if l == 0:
            print '********  Pavia Univ **********'
        elif l == 1:
            print '*******  Pavia Center *********'
        elif l == 2:
            print '*********  Salinas ************'
        data = d[l]
        print '                      Scale = ', scale
        try:
            np.mean(data.loc[data['scale']==scale,'Process_time'])
            print '               Process time = ', np.mean(data.loc[data['scale']==scale,'Process_time'])
        except:
            print '               Process time = ', 0
        print 'Ratio cut              time = ', np.mean(data.loc[data['scale']==scale,'Rcut_time'])
        print 'Ratio cut          Accuracy = ', np.mean(data.loc[data['scale']==scale,'accRcut'])
        print 'Power Ratio cut        time = ', np.mean(data.loc[data['scale']==scale,'PRcut_time'])
        print 'Power Ratio cut    Accuracy = ', np.mean(data.loc[data['scale']==scale,'accPRcut'])
        print '********************************'
