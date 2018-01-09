import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('results_6_i7.csv', header = None)
colnames = ['dim','size','process_time','Rcut_time','accRcut','AMIRcut','ARIRcut', 'PRcut_time','accPRcut','AMIPRcut','ARIPRcut']
data.columns = colnames

# data = data[data['dim']<=20]

plt.figure()

tmin = min(np.min(data['Rcut_time']), np.min(data['PRcut_time']) )
tmax = max(np.max(data['Rcut_time']), np.max(data['PRcut_time']) )
plt.axis([tmin, tmax, tmin, tmax])

plt.plot(np.arange(tmin,tmax,0.001),np.arange(tmin,tmax,0.001))
plt.scatter(data['Rcut_time'],data['PRcut_time'], color='b')

plt.xlabel('Ratio cut')
plt.ylabel('Power Ratio cut')
plt.savefig('./img/Example6_times.pdf', bbox_inches='tight')
plt.close('all')


plt.figure()

tmin = min(np.min(data['accRcut']), np.min(data['accPRcut']) )
tmax = max(np.max(data['accRcut']), np.max(data['accPRcut']) )
plt.axis([tmin, tmax, tmin, tmax])

plt.plot(np.arange(tmin,tmax,0.001),np.arange(tmin,tmax,0.001))

plt.scatter(data['accRcut'],data['accPRcut'], color='b')
plt.xlabel('Ratio cut')
plt.ylabel('Power Ratio cut')
plt.savefig('./img/Example6_Accuracy.pdf', bbox_inches='tight')
plt.close('all')


plt.figure()
tmin = min(np.min(data['ARIRcut']), np.min(data['ARIPRcut']) )
tmax = max(np.max(data['ARIRcut']), np.max(data['ARIPRcut']) )
plt.axis([tmin, tmax, tmin, tmax])

plt.plot(np.arange(tmin,tmax,0.001),np.arange(tmin,tmax,0.001))

plt.scatter(data['ARIRcut'],data['ARIPRcut'], color='b')
plt.xlabel('Ratio cut')
plt.ylabel('Power Ratio cut')
plt.savefig('./img/Example6_ARI.pdf', bbox_inches='tight')
plt.close('all')


plt.figure()
tmin = min(np.min(data['AMIRcut']), np.min(data['AMIPRcut']) )
tmax = max(np.max(data['AMIRcut']), np.max(data['AMIPRcut']) )
plt.axis([tmin, tmax, tmin, tmax])

plt.plot(np.arange(tmin,tmax,0.001),np.arange(tmin,tmax,0.001))
plt.scatter(data['AMIRcut'],data['AMIPRcut'], color='b')
plt.xlabel('Ratio cut')
plt.ylabel('Power Ratio cut')
plt.savefig('./img/Example6_AMI.pdf', bbox_inches='tight')
plt.close('all')


data['merge'] = [str(x)+"_"+str(y) for x,y in zip(data['dim'],data['size'])]
for dim in [10,15,20]:
    for thresh in list(np.unique(data['size'])):
        print '********************************************'
        print '              Dimension = ', dim
        print '                   Size = ', np.mean(data.loc[data['merge']== str(dim)+"_"+str(thresh),'size'])
        print 'time for preprocessing  = ', np.mean(data.loc[data['merge']== str(dim)+"_"+str(thresh),'process_time'])
        print 'Rcut  :      time_taken = ', np.mean(data.loc[data['merge']== str(dim)+"_"+str(thresh),'Rcut_time'])
        print '               Accuracy = ', np.mean(data.loc[data['merge']== str(dim)+"_"+str(thresh),'accRcut'])
        print '                    ARI = ', np.mean(data.loc[data['merge']== str(dim)+"_"+str(thresh),'ARIRcut'])
        print '                    AMI = ', np.mean(data.loc[data['merge']== str(dim)+"_"+str(thresh),'AMIRcut'])
        print 'PRcut :      time_taken = ', np.mean(data.loc[data['merge']== str(dim)+"_"+str(thresh),'PRcut_time'])
        print '               Accuracy = ', np.mean(data.loc[data['merge']== str(dim)+"_"+str(thresh),'accPRcut'])
        print '                    ARI = ', np.mean(data.loc[data['merge']== str(dim)+"_"+str(thresh),'ARIPRcut'])
        print '                    AMI = ', np.mean(data.loc[data['merge']== str(dim)+"_"+str(thresh),'AMIPRcut'])
        print '********************************************'
