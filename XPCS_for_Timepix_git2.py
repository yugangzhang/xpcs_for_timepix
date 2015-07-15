# -*- coding: utf-8 -*-
######################################################################################
##Revised Based on Yorick_Multitau_Code Obtained from Dr. Andrei Fluerasu###############
#####################Do one-teim correlation for TimePix#############################
####################   Coded by Dr. Yugang Zhang #################################
####################   631-885-4714   ############################################
####################   Email: yuzhang@bnl.gov     ####################################
########################################################################################
#################### or: yuzhangnew@icloud.com      ##################################
####################  At Brookhaven National Lab   #########################################
############################## July 10, 2015##############################################
#######################################################################################
######################################################################################
############################################################################################################
#######################################################################################
######################################################################################
################################################################################

from numpy import pi,sin,arctan,sqrt,mgrid,where,shape,exp,linspace,std,arange
from numpy import power,log,log10,array,zeros,ones,reshape,mean,histogram,round,int_
from numpy import indices,hypot,digitize,ma,histogramdd,apply_over_axes,sum
from numpy import around,intersect1d, ravel, unique,hstack,vstack,zeros_like
from numpy import save, load, dot
from numpy.linalg import lstsq
from numpy import polyfit,poly1d;
import sys
import pandas as pd
import matplotlib.pyplot as plt 
from Init_for_Timepix import * # the setup file 
import time
 

#########################################
T = True
F = False
 

def read_xyt_frame(  n=1 ):
    ''' Load the xyt txt files:
        x,y is the detector (x,y) coordinates
        t is the time-encoder (when hitting the detector at that (x,y))
        DATA_DIR is the data filefold path
        DataPref is the data prefix
        n is file number
        the data name will be like: DATA_DIR/DataPref_0001.txt
        return the histogram of the hitting event

    '''
    import numpy as np 
    ni = '%04d'%n
    fp = DATA_DIR +   DataPref + '%s.txt'%ni
    data = np.genfromtxt( fp, skiprows=0)[:,2] #take the time encoder
    td = np.histogram( data, bins= np.arange(11810) )[0] #do histogram    
    return td 
 
def readframe_series(n=1 ):
    ''' Using this universe name for all the loading fucntions'''
    return read_xyt_frame( n )
    

class xpcs( object):
    def __init__(self):
        """ DOCUMENT __init__(   )
        the initilization of the XPCS class
          """
        self.version='version_0'
        self.create_time='July_14_2015'
        self.author='Yugang_Zhang@chx11id_nsls2_BNL'
 
    def delays(self,time=1,
               nolevs=None,nobufs=None, tmaxs=None): 
        ''' DOCUMENT delays(time=)
        Using the lev,buf concept, to generate array of time delays
        return array of delays.
        KEYWORD:  time: scale delays by time ( should be time between frames)
                  nolevs: lev (a integer number)
                  nobufs: buf (a integer number)
                  tmax:   the max time in the calculation, usually, the noframes
                  
        '''
         
        if nolevs is None:nolevs=nolev #defined by the set-up file
        if nobufs is None:nobufs=nobuf #defined by the set-up file
        if tmaxs is None:tmaxs=tmax    #defined by the set-up file
        if nobufs%2!=0:print "nobuf must be even!!!"    
        dly=zeros( (nolevs+1)*nobufs/2 +1  )        
        dict_dly ={}
        for i in range( 1,nolevs+1):
            if i==1:imin= 1
            else:imin=nobufs/2+1
            ptr=(i-1)*nobufs/2+ arange(imin,nobufs+1)
            dly[ptr]= arange( imin, nobufs+1) *2**(i-1)            
            dict_dly[i] = dly[ptr-1]            
        dly*=time 
        dly = dly[:-1]
        dly_ = dly[: where( dly < tmaxs)[0][-1] +1 ]
        self.dly=dly 
        self.dly_=dly_
        self.dict_dly = dict_dly 
        return dly
 

    def make_qlist(self):
        ''' DOCUMENT make_qlist( )
        Giving the noqs, qstart,qend,qwidth, defined by the set-up file        
        return qradi: a list of q values, [qstart, ...,qend] with length as noqs
               qlist: a list of q centered at qradi with qwidth.
        KEYWORD:  noqs, qstart,qend,qwidth::defined by the set-up file 
        '''  
        qradi = linspace(qstart,qend,noqs)
        qlist=zeros(2*noqs)
        qlist[::2]= round(qradi-qwidth/2)  #render  even value
        qlist[1::2]= round(qradi+(1+qwidth)/2) #render odd value
        qlist[::2]= int_(qradi-qwidth/2)  #render  even value
        qlist[1::2]= int_(qradi+(1+qwidth)/2) #render odd value
        if qlist_!=None:qlist=qlist_
        return qlist,qradi

    def calqlist(self, qmask=None ,  shape='circle' ):
        ''' DOCUMENT calqlist( qmask=,shape=, )
        calculate the equvilent pixel with a shape,
        return
            qind: the index of q
            pixellist: the list of pixle
            nopr: pixel number in each q
            nopixels: total pixel number       
        KEYWORD:
            qmask, a mask file;
            qlist,qradi is calculated by make_qlist()        
            shape='circle', give a circle shaped qlist
            shape='column', give a column shaped qlist
            shape='row', give a row shaped qlist             
        '''
         
        qlist,qradi = self.make_qlist()
        y, x = indices( [dimy,dimx] )  
        if shape=='circle':
            y_= y- ceny +1;x_=x-cenx+1        
            r= int_( hypot(x_, y_)    + 0.5  )
        elif shape=='column': 
            r= x
        elif shape=='row':
            r=y
        else:pass
        r= r.flatten()         
        noqrs = len(qlist)    
        qind = digitize(r, qlist)
        if qmask is  None:
            w_= where( (qind)%2 )# qind should be odd;print 'Yes'
            w=w_[0]
        else:
            a=where( (qind)%2 )[0]            
            b=where(  mask.flatten()==False )[0]
            w= intersect1d(a,b)
        nopixels=len(w)
        qind=qind[w]/2
        pixellist= (   y*dimx +x ).flatten() [w]
        nopr,bins=histogram( qind, bins= range( len(qradi) +1 ))
        return qind, pixellist,nopr,nopixels

    ###########################################################################
    ########for one_time correlation function for xyt frames
    ##################################################################

    def autocor_xytframe(self, n):
        '''Do correlation for one xyt frame--with data name as n '''
        dly_ = xp.dly_ 
        #cal=0
        gg2=zeros( len( dly_) )
        data = read_xyt_frame( n )  #load data
        datm = len(data)
        for tau_ind, tau in enumerate(dly_):
            IP= data[: datm - tau]
            IF= data[tau: datm ]             
            gg2[tau_ind]=   dot( IP, IF )/ ( IP.mean() * IF.mean() * float( datm - tau) )

        return gg2
 

    def autocor(self, noframes=10):
        '''Do correlation for xyt file,
           noframes is the frame number to be correlated
        '''
        start_time = time.time() 
        for n in range(1,noframes +1 ): # the main loop for correlator
            gg2 = self.autocor_xytframe( n )
            if n==1:g2=zeros_like( gg2 ) 
            g2 += (  gg2 - g2 )/ float( n  ) #average  g2
            #print n
            if noframes>10: #print progress...
                if  n %( noframes / 10) ==0:
                    sys.stdout.write("#")
                    sys.stdout.flush()                
        elapsed_time = time.time() - start_time
        print 'Total time: %.2f min' %(elapsed_time/60.)        
        return g2

    
    def plot(self, y,x=None):
        '''a simple plot'''
        if x is None:x=arange( len(y))
        plt.plot(x,y,'ro', ls='-')
        plt.show()

 
    def g2_to_pds(self, dly, g2, tscale = None):
        '''convert g2 to a pandas frame'''        
        if len(g2.shape)==1:g2=g2.reshape( [len(g2),1] )
        tn, qn = g2.shape
        tindex=xrange( tn )
        qcolumns = ['t'] + [ 'g2' ]
        if tscale is None:tscale = 1.0
        g2t = hstack( [dly[:tn].reshape(tn,1) * tscale, g2 ])        
        g2p = pd.DataFrame(data=g2t, index=tindex,columns=qcolumns)         
        return g2p

    def show(self,g2p,title):
        t = g2p.t  
        N = len( g2p )
        ylim = [g2p.g2.min(),g2p[1:N].g2.max()]
        g2p.plot(x=t,y='g2',marker='o',ls='--',logx=T,ylim=ylim);
        plt.xlabel('time delay, ns',fontsize=12)
        plt.title(title)
        plt.savefig( RES_DIR + title +'.png' )       
        plt.show()
    
 

######################################################
 

xp=xpcs(); #use the xpcs class
dly = xp.delays()
if T:
    fnum = 100
    g2=xp.autocor( fnum )
    filename='g2_-%s-'%(fnum)
    save( RES_DIR + FOUT + filename, g2)
    ##g2= load(RES_DIR + FOUT + filename +'.npy')
    g2p = xp.g2_to_pds(dly,g2, tscale = 20)
    xp.show(g2p,'g2_run_%s'%fnum)
 
