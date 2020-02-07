# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:04:33 2017

@author: S Vinu Sankar

This EBCTPS.py contains 3 important function required for interpolate.py
The 3 functions are epoch(), ETS() and EPS()

1. epoch(x, Fs) - Function for splitting signal into frames (overlapping): one per row
2. ETS(x, Fs, scaling_factor, zff_gci) - Function that performs Epoch based Time Scaling
3. EPS(x, Fs, scaling_factor, zff_gci) - Function that performs Epoch based Pitch Scaling

The modules used are numpy and scipy.signal
http://www.numpy.org/
https://docs.scipy.org/doc/scipy/reference/signal.html
"""

import numpy as np
from scipy.signal import lfilter, resample


# Function for splitting signal into frames (overlapping): one per row
# enframe(x, np.hamming(n), n/4) will return a 75% overlapped hamming window of length n
# x is input signal
# win is window or window length in samples
# inc is frame increment in samples
def enframe(x, win, inc):
        
      nx = len(x)
      nwin = len(win)
            
      if nwin == 1:
            lw = win[0]
            w = np.ones(lw)
                
      else:   
            lw = nwin
            w = win
                
      nli = nx-lw+inc
      nf = int(np.abs(nli/inc)/(nli/inc) * np.floor(np.abs(nli/inc)))
      f = np.zeros((nf, lw))
      indf = inc*np.array(list(range(nf)))
      inds = np.array(list(range(1, lw+1)))
      a = np.array([[indf[i]]*lw for i in range(len(indf))], dtype = 'int64') + np.array(([list(inds)]*nf), dtype = 'int64')-1
      f = x[a]  
           
      if nwin > 1:
            f = f*np.array([list(w)]*nf)
          
      return np.array(f, dtype = 'float16')


# Function for finding out epoch in the given audio sample data
# epoch(x, Fs) returns an array with the positions of epochs in the signal
# x is input signal
# Fs is the sample rate of the signal
def epoch(x, Fs):
    global j,a,sp_mat,Rx   ,sp # epoch() starts
    if len(x.shape) > 1:
        x = np.array([x[i][0] for i in range(len(x))])
        
    L = len(x)
    s = np.zeros(L)
    s[0] = x[0]
    s[1: L] = x[1: L] - x[0: L-1]
    
    ak = [1, -2, 1]
    y1 = lfilter([1], ak, s)
    y2 = lfilter([1], ak, y1)
    
    W = int(np.floor(0.03 * Fs))
    hammW = np.array(np.hamming(W), dtype = 'float16')
    one_ms = int(np.floor(0.001 * Fs))

    # enframe performed on signal s    
    sp_mat = enframe(s, hammW, int(np.floor(W/2))).T
    ham_win = np.array(np.correlate(hammW, hammW, mode = 'full'), dtype = 'float64')
    ham_win /= max(ham_win)
    
    strt = int(np.floor(0.032*Fs))
    stpt = int(np.floor(0.045*Fs))
    ind = np.zeros(sp_mat.shape[1], dtype = 'float16')
    xax = np.zeros(14, dtype = 'float16')

    for j in range(1, sp_mat.shape[1]+1):
        
        sp = np.array([sp_mat[i][j-1] for i in range(len(sp_mat))])
        Rx = np.correlate(sp, sp, mode = 'full')
        
        if max(Rx) == 0:
              Rx = np.array([float('inf')]*len(Rx))
        
        else:
              Rx /= (max(Rx)*ham_win)
              
        a = Rx[strt-1: stpt]
        indm = list(a).index(max(a))
        ind[j-1] = indm + 2*one_ms
     
    for j in range(1, 15):
        
        xax[j-1] = (j+1)*one_ms
           
    pp, fr = np.histogram(ind, bins = xax)
    pp[0] += len(ind)-np.sum(pp)
    m_pitch_ind = list(pp).index(max(pp))
    
    N = int(np.floor(1.5*fr[m_pitch_ind]/2))
    
    #pitch = int(fr[m_pitch_ind]*1000/Fs)
    # This way pitch of the signal can be obtained
    
    y22 = list(np.zeros(N))
    y22.extend(list(y2))
    y22.extend(list(np.zeros(N)))
    y22 = np.array(y22)
    zfr_sig1 = np.zeros(len(y22))
    
    for j in [1,2,3]:
        
        fltavg = np.ones(2*N+1)/(2*N+1)
        yavg = np.convolve(y22, fltavg)
        lst = int((len(yavg)-len(y22))/2)
        yavg = yavg[-(lst+len(y22)): -lst]
        zfr_sig1[N: len(y22)-N] = y22[N: len(y22)-N] - yavg[N: len(yavg)-N]
        y22 = zfr_sig1
    
    zfr_sig = y22[N: len(y22)-N]
    zff_gci1 = [0]
    zff_gci1.extend(list(np.diff(np.sign(zfr_sig)/2)))
    zff_gci1 = np.array(zff_gci1, dtype = 'int16')
    zc_ind = np.array([i for i in range(len(zff_gci1)) if zff_gci1[i]>0])
    zff_gci = np.zeros(L)
    zff_gci[zc_ind] = 1
    
    return zff_gci

# Function that performs Epoch based Time Scaling
# ETS(x, Fs, scaling_factor, zff_gci) returns the scaled audio signal
# x is input signal
# Fs is the sample rate of the input signal
# scaling_factor is the factor by which the audio is to be time scaled
# zff_gci is an array that contains epoch points
# zff_gci is obtained using the epoch() function
def ETS(x, Fs, scaling_factor, zff_gci):
    
    if scaling_factor == 1:
        return x
    
    else:
        
        L = len(x)
        
        zff_array = zff_gci
        frame_len = int(np.floor(0.02*Fs))
        scaling_factor = 1/scaling_factor
        shift_val = int(np.floor(frame_len/2))
        hamming_win = np.hamming(2*shift_val)
        Ss = frame_len - shift_val
        output_len = round(L*scaling_factor)
        
        ylast = shift_val
        zff_array_y = np.zeros(output_len)
        scaled_sp = np.zeros(output_len)
        Kmax = round(0.01*Fs)
        
        zff_array = np.concatenate((np.zeros(shift_val), zff_array, np.zeros(Kmax+shift_val)), axis = 0)
        x = np.concatenate((np.zeros(shift_val), x, np.zeros(Kmax+shift_val)), axis = 0)
        zff_array_y[: shift_val] = zff_array[: shift_val]
        scaled_sp[: shift_val] = x[: shift_val]
        
        i = 0
        
        km = []
        
        # main loop performing the scaling part
        for y_last in range(shift_val, output_len - frame_len + 1, Ss):
            
            search_strt_pt_Sa = round(y_last/scaling_factor)
            search_end_pt_Sa = search_strt_pt_Sa + frame_len - 1
            search_strt_pt_y = ylast - shift_val +1
            search_end_pt_y = ylast
            
            mat_ind_Sa = [j for j in range(len(zff_array[search_strt_pt_Sa-1: search_end_pt_Sa])) if zff_array[j+search_strt_pt_Sa-1]>0]
            mat_ind_y = [j for j in range(len(zff_array_y[search_strt_pt_y-1: search_end_pt_y])) if zff_array_y[j+search_strt_pt_y-1]]
            
            i += 1
            
            if mat_ind_Sa == [] or mat_ind_y == [] or (mat_ind_Sa[0] - mat_ind_y[0]) > Kmax:
                km.append(0)
            
            else:
                valid_ind = [j for j in range(len(mat_ind_Sa)) if mat_ind_Sa[j]>mat_ind_y[0]]
                
                if valid_ind == []:
                    km.append(0)
                
                else:
                    km.append(mat_ind_Sa[valid_ind[0]] - mat_ind_y[0])
            
            y_ovrlp = np.array(range(ylast-shift_val+1, ylast+1), dtype = 'int64')
            y_nonovrlp = np.array(range(ylast+1, ylast+Ss+1), dtype = 'int64')
            x_ovrlp = np.array(range(search_strt_pt_Sa+km[i-1], search_strt_pt_Sa+shift_val+km[i-1]), dtype = 'int64')
            x_nonovrlp = np.array(range(search_strt_pt_Sa+km[i-1]+shift_val, search_end_pt_Sa+km[i-1]+1), dtype = 'int64')
            
            zff_array_y[y_ovrlp-1] = zff_array_y[y_ovrlp-1] + zff_array[x_ovrlp-1]
            zff_array_y[y_nonovrlp-1] = zff_array[x_nonovrlp-1]
            
            scaled_sp[y_ovrlp-1] = scaled_sp[y_ovrlp-1]*hamming_win[shift_val:] + x[x_ovrlp-1]*hamming_win[: shift_val]
            scaled_sp[y_nonovrlp-1] = x[x_nonovrlp-1]
            ylast += Ss 
                    
        return scaled_sp

# Function that performs Epoch based Pitch Scaling
# EPS(x, Fs, scaling_factor, zff_gci) returns the  pitch scaled audio signal
# x is input signal
# Fs is the sample rate of the input signal
# scaling_factor is the factor by which the audio is to be pitch scaled
# zff_gci is an array that contains epoch points
# zff_gci is obtained using the epoch() function
def EPS(x, Fs, scaling_factor, zff_gci):

    if scaling_factor == 1:
        return x
    
    else:    
        
        time_scaled_sp = ETS(x, Fs, 1/scaling_factor, zff_gci)
        scaled_sp = resample(time_scaled_sp, int(len(time_scaled_sp)/scaling_factor))
        
        return scaled_sp