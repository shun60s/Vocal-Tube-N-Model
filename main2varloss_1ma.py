#coding:utf-8

#
# variable two tube model with loss, draw waveform, considering glottal voice source processed lpf, and mouth radiation
#                          save generated waveform as a wav file

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from scipy.io.wavfile import write as wavwrite
from twotubevariable_loss import *
from glottal_filtering import *  # use source processed variable low pass filter
from HPF import *
from tube_1MA import *


# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 
#  matplotlib  2.1.1
#  scipy 1.0.0


def plot_freq_res(twotube, label, glo, hpf):
	plt.xlabel('Hz')
	plt.ylabel('dB')
	plt.title(label)
	amp0, freq=glo.H0(freq_high=5000, Band_num=256)
	amp1, freq=twotube.H0(freq_high=5000, Band_num=256)
	amp2, freq=hpf.H0(freq_high=5000, Band_num=256)
	plt.plot(freq, (amp0+amp1+amp2))

def plot_waveform(twotube, label, glo, hpf):
	# you can get longer input source to set bigger repeat_num 
	yg_repeat=glo.yg_lpf
	y2tm=twotube.process(yg_repeat)
	yout=hpf.iir1(y2tm)
	# draw waveform subplot(2,1,1)
	plt.subplot(2,1,1)
	plt.xlabel('mSec')
	plt.ylabel('level')
	plt.title( label )
	plt.plot( (np.arange(len(yout)) * 1000.0 / glo.sr) , yout)
	# draw loss subplot(2,1,2)
	plt.subplot(2,1,2)
	plt.xlabel('mSec')
	plt.ylabel('loss')
	plt.title( 'loss per tube (blue: A1,L1  green: A2,L2)' )
	plt.plot( (np.arange(len(twotube.A1_loss_list)) * 1000.0 / glo.sr) , twotube.A1_loss_list, color='b')
	plt.plot( (np.arange(len(twotube.A2_loss_list)) * 1000.0 / glo.sr) , twotube.A2_loss_list, color='g')
	#
	# Loss curve may have stepwise changes.
	# This cause is low resolution to digitalize tube length.
	# If sample_rate is higher, stepwise is smaller. But, it also needs to adjust passing_loss_ratio.
	#
	return yout

def erase_head(yin, glo, n_cycle=1):
	# erase initial transient head
	
	tclosed= glo.tclosed # 5.0 msec
	trise=   glo.trise   # 6.0 msec
	tfall=   glo.tfall   # 2.0 msec
	sr=      glo.sr # sampling_rate # 48000
	
	one_cycle= tclosed + trise + tfall
	one_cycle_steps= int((one_cycle / 1000.0) * sr)
	erase_steps= one_cycle_steps * n_cycle
	print ('one cycle step ', one_cycle_steps)
	print ('erase initial portion of time [msec]', (erase_steps / sr ) * 1000.0)
	
	if len(yin) > erase_steps:
		yout=np.copy( yin[ erase_steps:])
	else:
		print ('ERROR: data size is too small. def erase_head in main2var_ua.py')
		yout= np.zeros(1)
	
	return yout, erase_steps
	
def append_zero_data(yi, append_zero_data_length=100, sampling_rate=48000):
	# input append_zero_data_length is append zero time duration to RESP0, unit is [msec]
	append_zero_data_n0= int( (append_zero_data_length / 1000.0) * sampling_rate)
	if append_zero_data_n0 > 0:
		y0= np.zeros(append_zero_data_n0)
	return np.concatenate( (y0, yi, y0) )

def save_wav( yout, wav_path, sampling_rate=48000):
	wavwrite( wav_path, sampling_rate, ( yout * 2 ** 15).astype(np.int16))
	print ('save ', wav_path) 

if __name__ == '__main__':
	
	import itertools
	
	# set variables
	passing_loss_ratio = [0.01]  # [0.01] # vary loss ratio
	standard_area=[2.0] # vary standard area
	n_cycle_list=[1]  #  [2, 1] # vary erase head cycle
	
	clist=list(itertools.product(passing_loss_ratio,standard_area, n_cycle_list))
	print (clist)
	
	# instance
	glo=Class_Glottal_LPF(repeat_num=15) # sampling_rate=96000), need change passing_loss_ratio to half
	glo.plot_waveform()
	start_time= (glo.sp / glo.sr) * 1000.0
	print ('start time', start_time)
	
	A=Class_CMA2MA(transient_time=150, start_hold_time=start_time, tc=0.1, sampling_rate=glo.sr)  # simulated opening mouth
	A.draw_cross_section_area()
	A.f_show_all()
	
	# instance for mouth radiation effect
	hpf=Class_HPF()       
	
	
	for i in range (len(clist)):
		#
		label='ma'
		# instance variable two tube model with loss
		tube=Class_TwoTube_variable_loss( A, passing_loss_ratio = clist[i][0],  standard_area=2.0, sampling_rate=glo.sr )
		# draw
		fig = plt.figure()
		#
		plt.subplot(2,1,1)
		yout_ua=plot_waveform(tube, 'Waveform ' + label + ' (using green portion: 1a' + str(clist[i][2]) + ' ) loss ratio: ' + str(clist[i][0]) + ' std area: ' + str(clist[i][1])  , glo, hpf)
		
		# remove initial transient portion per pitch period
		plt.subplot(2,1,1) # re-draw waveform subplot(2,1,1) again
		yout_ua2, erase_steps= erase_head(yout_ua, glo, n_cycle= clist[i][2])
		plt.plot( ((np.arange(len(yout_ua2)) + + erase_steps) * 1000.0 / glo.sr) , yout_ua2,  color='g')
		save_wav(yout_ua2, 'yout_' + label + '_1a' + str(clist[i][2]) + '_varloss' + str(clist[i][0]).replace('.','_') + '_stdarea' + str(clist[i][1]).replace('.','_') + '.wav', sampling_rate=glo.sr)  # save generated wa + 'veform as a wav file
		#
		yout_ua2_zero=append_zero_data(yout_ua2, sampling_rate=glo.sr)
		save_wav(yout_ua2_zero, 'yout_' + label + '_1a' + str(clist[i][2]) + '_varloss' + str(clist[i][0]).replace('.','_') + '_stdarea' + str(clist[i][1]).replace('.','_') + '_long.wav', sampling_rate=glo.sr)  # save generated waveform as a wav file to listen
		
		fig.tight_layout()
		plt.show()
	
#This file uses TAB

