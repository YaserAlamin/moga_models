"""
Input arguments:
Number of models

This code should be placed in the same folder that output models from moga and the original dataset that we provide for moga exist
The output models from moga should name by integer numbers like 1.npz, 2.npz,..., maxNumberOfModels.npz
Output:
The code produces three txt files as output:
R.txt which calculates FP,FN,F for train, test, validation and whole dataset as well as model complexity for each model
best Models_classification result.txt which calculates FP,FN,F for train, test, validation and whole dataset as well as model complexity for
best models.
Best models are models whose number of Fs in at least one of train, test, validation or the whole dataset is equal to minimum
Features of best Models.txt which shows which features are used within the best model set. 
"""
from numpy import *
from numpy import load,save
from scipy.io import savemat
import sys
import math
import csv
import gc


def rbf_fn(c,sig,x):# produce the output of all data samples for one neuron
    
    size_x=x.shape
    f=zeros((size_x[0],1),dtype=double)  
  for i in range(size_x[0]):
    #print "x",x[i,:]
    #print"c",c
    const=(-1)/(2*(sig**2))
    sum=0
    sum=linalg.norm(x[i,:]-c)
    
    sum=sum**2
    #sum=sum/(2*(sig**2))
    sum=const*sum
    f[i,0]=exp(sum)
  #print "======================"
  #print f.shape  
  return f
  
def f1(filename,ds2):
  ds=load(filename)
  spreads=ds['spreads']
  weights=ds['weights']
  features=sort(ds['features'])
  centers=ds['centers']
  #print "features",features,"centers",centers


  
  """V=ds2['notLBP_V']
  TR=ds2['notLBP_TR']
  TE=ds2['notLBP_TE']"""
  V=ds2['VA_SET']
  TR=ds2['TR_SET']
  TE=ds2['TE_SET']

  DS=concatenate((V,TE), axis=0)
  DS=concatenate((DS,TR), axis=0)
  size_DS=DS.shape
  size_c=centers.shape
  size_V=V.shape
  size_TR=TR.shape
  size_TE=TE.shape
  ModelComplexity=size_c[0]*size_c[1]
  #-----------------------------------------------whole data set
  a_ds=ones((size_DS[0],size_c[0]+1),dtype=double)
  #print "a",a.shape
  for k in range(0,size_c[0]):
    bbds=rbf_fn(centers[k,:],spreads[k,0],DS[:,features])
    #print "bb", bb.shape
    a_ds[:,k]=bbds[:,0]

  
  Y_ds=dot(a_ds,weights)
  RMSE_ds=sqrt(sum(power((Y_ds[:,0]-DS[:,0]),2))/size_DS[0])
  print "whole dataset: Norm_y", linalg.norm(Y_ds)
  print "RMSE whole dataset: ", RMSE_ds
  FPds=0;
  FNds=0;
  for i in range(0,size_DS[0]):
    if (Y_ds[i,0]>0) and (DS[i,0]==-1):
      FPds=FPds+1
    else:
      if (Y_ds[i,0]<0) and (DS[i,0]==1):
        FNds=FNds+1
  print "FPds",FPds,"FNds",FNds
 
 
  #-----------------------------------------------Validation set
  a_v=ones((size_V[0],size_c[0]+1),dtype=double)
  #print "a",a.shape
  for k in range(0,size_c[0]):
    bbv=rbf_fn(centers[k,:],spreads[k,0],V[:,features])
    #print "bb", bb.shape
    a_v[:,k]=bbv[:,0]

  
  Y_v=dot(a_v,weights)
  RMSE_v=sqrt(sum(power((Y_v[:,0]-V[:,0]),2))/size_V[0])
  Min_v=min(V[:,0]-Y_v[:,0])
  Avg_v=mean(V[:,0]-Y_v[:,0])
  Max_v=max(V[:,0]-Y_v[:,0])
  print "validation set: Norm_y", linalg.norm(Y_v)
  print "RMSE validation set: ", RMSE_v
  FPv=0;
  FNv=0;
  for i in range(0,size_V[0]):
    if (Y_v[i,0]>0) and (V[i,0]==-1):
      FPv=FPv+1
    else:
      if (Y_v[i,0]<0) and (V[i,0]==1):
        FNv=FNv+1
  print "FPv",FPv,"FNv",FNv
 
  #-----------------------------------------------Training set
  a_tr=ones((size_TR[0],size_c[0]+1),dtype=double)
  #print "a",a.shape
  for k in range(0,size_c[0]):
    #print "features",features.shape,"centers",centers.shape,"spreads",spreads.shape
    bbtr=rbf_fn(centers[k,:],spreads[k,0],TR[:,features])
    #print "bb", bb.shape
    a_tr[:,k]=bbtr[:,0]

  
  Y_tr=dot(a_tr,weights)
  RMSE_tr=sqrt(sum(power((Y_tr[:,0]-TR[:,0]),2))/size_TR[0])
  Min_tr=min(TR[:,0]-Y_tr[:,0])
  Avg_tr=mean(TR[:,0]-Y_tr[:,0])
  Max_tr=max(TR[:,0]-Y_tr[:,0])

  print "training set: Norm_y", linalg.norm(Y_tr)
  print "RMSE training set: ", RMSE_tr
  FPtr=0;
  FNtr=0;
  for i in range(0,size_TR[0]):
    if (Y_tr[i,0]>0) and (TR[i,0]==-1):
      FPtr=FPtr+1
    else:
      if (Y_tr[i,0]<0) and (TR[i,0]==1):
        FNtr=FNtr+1
  print "FPtr",FPtr,"FNtr",FNtr
  
  #-----------------------------------------------Test set
  a_tt=ones((size_TE[0],size_c[0]+1),dtype=double)
  #print "a",a.shape
  for k in range(0,size_c[0]):
    bbtt=rbf_fn(centers[k,:],spreads[k,0],TE[:,features])
    #print "bb", bb.shape
    a_tt[:,k]=bbtt[:,0]

  
  Y_tt=dot(a_tt,weights)
  RMSE_tt=sqrt(sum(power((Y_tt[:,0]-TE[:,0]),2))/size_TE[0])
  Min_tt=min(TE[:,0]-Y_tt[:,0])
  Avg_tt=mean(TE[:,0]-Y_tt[:,0])
  Max_tt=max(TE[:,0]-Y_tt[:,0])

  print "test set: Norm_y", linalg.norm(Y_tt)
  print "RMSE test set: ", RMSE_tt
  #print Y_tt[:,0],TE[:,0],(Y_tt[:,0]-TE[:,0])

  FPtt=0;
  FNtt=0;
  for i in range(0,size_TE[0]):
    #print Y_tt[i,0],TE[i,0]
    if (Y_tt[i,0]>0) and (TE[i,0]==-1):
      FPtt=FPtt+1
    else:
      if (Y_tt[i,0]<0) and (TE[i,0]==1):
        FNtt=FNtt+1
  print "FPtt",FPtt,"FNtt",FNtt
  
  #return Y_v,Y_tr,Y_tt
  #ds=None
  #del ds
  ds.close()
  gc.collect()
  return FPtr,FNtr,FPtt,FNtt,FPv,FNv,FPds,FNds,RMSE_tr,RMSE_tt,ModelComplexity,linalg.norm(weights),RMSE_v,Min_v,Avg_v,Max_v,Min_tr,Avg_tr,Max_tr,Min_tt,Avg_tt,Max_tt
   
if __name__=='__main__':
  #Y_v,Y_tr,Y_tt=f1()
  #ds2=load('data_notLBP_new_43_f.npz')
  #ds2=load('data_scenario1_convexHull_WithOut_Target.npz')
  #ds2=load('data_cvh_scn_5.npz')
  ds2=load('MOGA_DATA_136i_os.npz')
  n=int(sys.argv[1]) # number of models 
  #NumberOfFeatures=ds2['notLBP_V'].shape[1]-1
  NumberOfFeatures=ds2['VA_SET'].shape[1]-1
  R=zeros((n,27),dtype=float)
  for i in range(n):
    filename=str(i+1)+'.npz'
    FPtr,FNtr,FPtt,FNtt,FPv,FNv,FPds,FNds,RMSE_tr,RMSE_tt,ModelComplexity,Norm_weights,RMSE_v,Min_v,Avg_v,Max_v,Min_tr,Avg_tr,Max_tr,Min_tt,Avg_tt,Max_tt=f1(filename,ds2)
    R[i,0]=i+1
    R[i,1]=FPtr
    R[i,2]=FNtr
    R[i,3]=FPtr+FNtr
    R[i,4]=FPtt
    R[i,5]=FNtt
    R[i,6]=FPtt+FNtt
    R[i,7]=FPv
    R[i,8]=FNv
    R[i,9]=FPv+FNv
    R[i,10]=FPds
    R[i,11]=FNds
    R[i,12]=FPds+FNds
    R[i,13]=ModelComplexity
    R[i,14]=RMSE_tr
    R[i,15]=RMSE_tt
    R[i,16]=RMSE_v
    R[i,17]=Norm_weights
    
    R[i,18]=Min_v
    R[i,19]=Avg_v
    R[i,20]=Max_v
    R[i,21]=Min_tr
    R[i,22]=Avg_tr
    R[i,23]=Max_tr
    R[i,24]=Min_tt
    R[i,25]=Avg_tt
    R[i,26]=Max_tt
    
    
  with open('R.txt', 'wb') as f:
    f.write(b'Model No.\tFPtr\tFNtr\tFPtr+FNtr\tFPtt\tFNtt\tFPtt+FNtt\tFPv\tFNv\tFPv+FNv\tFPds\tFNds\tFPds+FNds\tModelComplexity\tRMSE_tr\tRMSE_tt\tRMSE_v\tNorm_weights\tMin_v\tAvg_v\tMax_v\tMin_tr\tAvg_tr\tMax_tr\tMin_tt\tAvg_tt\tMax_tt\n')
    savetxt(f,R, delimiter="\t", fmt="%f")
  #-------------------------------------------------find features and their frequecy in best model set start
  
  Feature_allBestModels=zeros((0,NumberOfFeatures+1),dtype=int)
  classification_result_All_Best_models=zeros((0,27),dtype=float)
  classification_result_each_Best_model=zeros((1,27),dtype=float)
  Min_F_tr=min(R[:,3])
  Min_F_tt=min(R[:,6])
  Min_F_v=min(R[:,9])
  Min_F_ds=min(R[:,12])
  for i in range(n):
    if R[i,3]==Min_F_tr or R[i,6]==Min_F_tt or R[i,9]==Min_F_v or R[i,12]==Min_F_ds:
      classification_result_each_Best_model[0,0]=i+1 # Model number
      classification_result_each_Best_model[0,1]=R[i,1]
      classification_result_each_Best_model[0,2]=R[i,2]
      classification_result_each_Best_model[0,3]=R[i,3]
      classification_result_each_Best_model[0,4]=R[i,4]
      classification_result_each_Best_model[0,5]=R[i,5]
      classification_result_each_Best_model[0,6]=R[i,6]
      classification_result_each_Best_model[0,7]=R[i,7]
      classification_result_each_Best_model[0,8]=R[i,8]
      classification_result_each_Best_model[0,9]=R[i,9]
      classification_result_each_Best_model[0,10]=R[i,10]
      classification_result_each_Best_model[0,11]=R[i,11]
      classification_result_each_Best_model[0,12]=R[i,12]
      classification_result_each_Best_model[0,13]=R[i,13]
      classification_result_each_Best_model[0,14]=R[i,14]
      classification_result_each_Best_model[0,15]=R[i,15]
      classification_result_each_Best_model[0,16]=R[i,16]
      classification_result_each_Best_model[0,17]=R[i,17]
      
      classification_result_each_Best_model[0,18]=R[i,18]
      classification_result_each_Best_model[0,19]=R[i,19]
      classification_result_each_Best_model[0,20]=R[i,20]
      classification_result_each_Best_model[0,21]=R[i,21]
      classification_result_each_Best_model[0,22]=R[i,22]
      classification_result_each_Best_model[0,23]=R[i,23]
      classification_result_each_Best_model[0,24]=R[i,24]
      classification_result_each_Best_model[0,25]=R[i,25]
      classification_result_each_Best_model[0,26]=R[i,26]

      
      Feature_eachModel=zeros((1,NumberOfFeatures+1),dtype=int)
      tmp=load(str(i+1)+'.npz')
      Feature_eachModel[0,0]=i+1 # Model Number
      indx=tmp['features'].reshape(1,tmp['features'].shape[0])
      for y in range(indx.shape[1]):
    Feature_eachModel[0,indx[0,y]]=Feature_eachModel[0,indx[0,y]]+1
    
      Feature_allBestModels=concatenate((Feature_allBestModels,Feature_eachModel),axis=0)
      classification_result_All_Best_models=concatenate((classification_result_All_Best_models,classification_result_each_Best_model),axis=0)
      
  Frequency=sum(Feature_allBestModels,axis=0)
  attend_rate_featuresInBestModels=(sum(Feature_allBestModels,axis=0)*100)/Feature_allBestModels.shape[0]
  attend_rate_featuresInBestModels=attend_rate_featuresInBestModels.reshape(1,NumberOfFeatures+1)
  attend_rate_featuresInBestModels=attend_rate_featuresInBestModels[:,1:NumberOfFeatures+1]
  Feature_allBestModels=concatenate((Feature_allBestModels,Frequency.reshape(1,NumberOfFeatures+1)),axis=0)# add frequency of feature in last line
  Feature_allBestModels[Feature_allBestModels.shape[0]-1,0]=0#Model number column does not have sum
  
  header='Model No.'
  for i in range(NumberOfFeatures):
    header=header+'\tF'+str(i+1)
  header=header+'\n'
  with open('Features of best Models.txt', 'wb') as f2:
    
    f2.write(header)
    savetxt(f2,Feature_allBestModels[:,0:NumberOfFeatures+1], delimiter="\t", fmt="%f") 

  header_attend_rate='F1'
  for i in range(1,NumberOfFeatures):
    header_attend_rate=header_attend_rate+'\tF'+str(i+1)
  header_attend_rate=header_attend_rate+'\n'

  with open('AttendRateOfFeaturesInBestModels.txt', 'wb') as f4:
    
    f4.write(header_attend_rate)
    savetxt(f4,attend_rate_featuresInBestModels, delimiter="\t", fmt="%f") 


  with open('best Models_classification result.txt', 'wb') as f3:
    f3.write(b'Model No.\tFPtr\tFNtr\tFPtr+FNtr\tFPtt\tFNtt\tFPtt+FNtt\tFPv\tFNv\tFPv+FNv\tFPds\tFNds\tFPds+FNds\tModelComplexity\tRMSE_tr\tRMSE_tt\tRMSE_v\tNorm_weights\tMin_v\tAvg_v\tMax_v\tMin_tr\tAvg_tr\tMax_tr\tMin_tt\tAvg_tt\tMax_tt\n')
    savetxt(f3,classification_result_All_Best_models, delimiter="\t", fmt="%f")
   
  
  #-------------------------------------------------find features and their frequecy in best model set end
  
  print "min FPtr :",min(R[:,1])
  print "min FNtr :",min(R[:,2])
  print "min FPtt :",min(R[:,4])
  print "min FNtt :",min(R[:,5])
  print "min Model Complexity :",min(R[:,13])
  print "min RMSE_tr :",min(R[:,14])
  print "min RMSE_tt :",min(R[:,15])
  print "min Norm of weights :",min(R[:,17])
  
  print "mean FPtr :",mean(R[:,1])
  print "mean FNtr :",mean(R[:,2])
  print "mean FPtt :",mean(R[:,4])
  print "mean FNtt :",mean(R[:,5])
  print "mean Model Complexity :",mean(R[:,13])
  print "mean RMSE_tr :",mean(R[:,14])
  print "mean RMSE_tt :",mean(R[:,15])
  print "mean Norm of weights :",mean(R[:,17])

  
  print "max FPtr :",max(R[:,1])
  print "max FNtr :",max(R[:,2])
  print "max FPtt :",max(R[:,4])
  print "max FNtt :",max(R[:,5])
  print "max Model Complexity :",max(R[:,13])
  print "max RMSE_tr :",max(R[:,14])
  print "max RMSE_tt :",max(R[:,15])
  print "max Norm of weights :",max(R[:,17])


    #print Y_v,Y_tr,Y_tt
    
    