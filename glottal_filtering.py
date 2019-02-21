#coding:utf-8

# glottal voice source as input of Two Tubes Model of vocal tract
# that is processed variable low pass filter of which cut-off frequency changes per cycle time as below
#
#        frequency:
#         cut-off
#          fc2               ----------->
#                          ->
#                        ->
#         cut-off      ->
#          fc1 ------->
#                     |     |
#   cycle time:     scyc   ecyc
#   cut-off frequency changing position, scyc and ecyc

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0
#  scipy 1.0.0
#  matplotlib 2.1.1

import math
import matplotlib.pyplot as plt
import numpy as np
from glottal_fade_in import *
from scipy import signal
from scipy.io.wavfile import write as wavwrite


class Class_Glottal_LPF(object):
    def __init__(self,fc1=100.0, fc2=1000.0, btype='low', n_order=3, repeat_num=15, scyc=3, ecyc=7, sampling_rate=48000):
        # initial
        self.sr = sampling_rate
        self.repeat_num= repeat_num
        self.tclosed=5.0
        self.trise=6.0
        self.tfall=2.0
        self.glo= Class_Glottal(tclosed=self.tclosed, trise=self.trise, tfall=self.tfall, sampling_rate=self.sr, fade_in_cycle=0, tc=1.0)
        gain0=0.9 # multiply some constant value to avoid overflow at fixed 16bit data
        self.yg_repeat_fade_in=self.glo.make_N_repeat(repeat_num= self.repeat_num) * gain0 
        
        self.fc1= fc1
        self.fc2= fc2 # fc2 should be >= fc1
        self.scyc= scyc
        self.ecyc= ecyc
        self.sp= int(scyc * self.glo.LL)
        self.ep= int(ecyc * self.glo.LL) # ecyc should be >= scyc
        
        # check
        if self.sp > len(self.yg_repeat_fade_in):
            self.sp= len(self.yg_repeat_fade_in)
            print ('warning: self.sp was set to len(xin)')
        if self.ep > len(self.yg_repeat_fade_in):
            self.ep= len(self.yg_repeat_fade_in)
            print ('warning: self.ep was set to len(xin)')
        if self.ep < self.sp:
            self.ep= self.sp
            print ('warning: self.ep was set to self.sp')
        #print ('sp, ep ', self.sp, self.ep)
        self.make_fc_curve(self.yg_repeat_fade_in)
        
        self.n_order= n_order
        self.btype= btype
        
        # output is signal after filtering 
        self.filtering( self.yg_repeat_fade_in )
        self.output= np.copy(self.yg_lpf)
    
    def make_fc_curve(self, xin):
        self.fc_list=np.zeros( len (xin))
        for i in range( len(xin)):
            if i < self.sp:
                self.fc_list[i]= self.fc1
            elif i < self.ep:
                # linear curve
                self.fc_list[i]= ((self.fc2 - self.fc1)/(self.ep - self.sp)) * (i - self.sp) + self.fc1 
            else:
                self.fc_list[i]=self.fc2
    
    def filtering(self, xin):
        # compute fc1 duration
        xin1=xin[0 : self.sp]
        b1, a1= signal.butter(self.n_order, (self.fc1 / (self.sr/2.0)) , btype=self.btype, analog=False, output='ba') # b=Numerator(bunsi), a=denominator(bunbo)
        zi= signal.lfilter(b1, a1, np.zeros(len(a1)-1) )
        xout1, zf = signal.lfilter(b1, a1, xin1, zi=zi)
        
        # compute varing fc per step
        xout2=np.zeros( self.ep - self.sp )
        for i in range( self.sp, self.ep):
            b, a= signal.butter(self.n_order, (self.fc_list[i] / (self.sr/2.0)) , btype=self.btype, analog=False, output='ba') # b=Numerator(bunsi), a=denominator(bunbo)
            y, zf = signal.lfilter(b, a, [xin[i]], zi=zf)
            xout2[i - self.sp]=y[0]
        
        # compute fc2 duration
        if self.ep < len(xin):
            xin3=xin[self.ep:]
            b2, a2= signal.butter(self.n_order, (self.fc2 / (self.sr/2.0)) , btype=self.btype, analog=False, output='ba') # b=Numerator(bunsi), a=denominator(bunbo)
            xout3, zf = signal.lfilter(b2, a2, xin3, zi=zf)
        else:
            xout3=np.zeros(0)
        
        # combine 
        self.yg_lpf= np.concatenate( (xout1, xout2, xout3) )
        
        
    def append_zero_data(self, append_zero_data_length=0):
        # input append_zero_data_length is append zero time duration to RESP0, unit is [msec]
    	self.append_zero_data_n0= int( (append_zero_data_length / 1000.0) * self.sr)
    	if self.append_zero_data_n0 > 0:
    	    y0= np.zeros(self.append_zero_data_n0)
    	    self.output= np.concatenate( (y0, self.output, y0) )
        
    def plot_waveform(self,):
        # plot waveform
        fig = plt.figure()
        plt.subplot(3,1,1)
        plt.xlabel('mSec')
        plt.ylabel('level')
        plt.title('waveform (LPF input)')
        plt.plot( (np.arange(len(self.yg_repeat_fade_in )) * 1000.0 / self.sr) , self.yg_repeat_fade_in )
        
        plt.subplot(3,1,2)
        plt.xlabel('mSec')
        plt.ylabel('level')
        plt.title('waveform (LPF output)')
        plt.plot( (np.arange(len(self.yg_lpf)) * 1000.0 / self.sr) , self.yg_lpf)
        
        plt.subplot(3,1,3)
        plt.xlabel('mSec')
        plt.ylabel('frequency')
        plt.title('LPF cut-off frequency ')
        plt.plot( (np.arange(len(self.fc_list)) * 1000.0 / self.sr) , self.fc_list)
        
        fig.tight_layout()
        plt.show()
        

    def save_wav(self, label, wav_path=None):
        # save yout2 as a wav file
        if wav_path is None:
            wav_path = 'glottl_lpf_' + label + '.wav'
        wavwrite( wav_path, self.sr, ( self.output * 2 ** 15).astype(np.int16))
        print ('save ', wav_path) 

if __name__ == '__main__':
	
    # instance
    glo=Class_Glottal_LPF()
    
    glo.plot_waveform()
    
    glo.append_zero_data(100) # option
    glo.save_wav('long')
