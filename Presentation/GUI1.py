# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 06:07:54 2017

@author: S Vinu Sankar

This python file, GUI.py contains all the code for the GUI
and splitting the audio sample for processing and then joining the 
processed fragments later by overlapping.

This program would get an input graph from the user.
The GUI has a graph using matplotlib that plots cubic B-spline with 25 control
points of Input scale vs Output scale. The user can click over the point to 
adjust them. 

The program splits up the audio into fragmnets of 1 second length and processes
them according to the scales it obtains from the scale, and then combines them
using hamming window overlapping.

The user can decide whether to perform time scaling or pitch scaling.
The user can also perform these functions on a recorded audio using the GUI.
The processed audio will be played and the user will have the option to save
the audio file.

The modules used are matplotlib, tkinter, PIL, time, soundfile, sounddevice, 
numpy, scipy, math, os, EBCTPS(user-defined)

GITHUB repository: https://github.com/vinusankars/Time-and-Pitch-Scaling-with-Python

http://www.numpy.org/
https://docs.scipy.org/doc/
https://matplotlib.org/
https://docs.python.org/2/library/tkinter.html
http://www.pythonware.com/products/pil/
https://docs.python.org/2/library/time.html
https://python-sounddevice.readthedocs.io/en/0.3.7/
https://pypi.python.org/pypi/SoundFile/0.8.1
https://docs.python.org/2/library/math.html
https://docs.python.org/2/library/os.html
EBCTPS.py
"""

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib as mpl
import tkinter as tk
import tkinter.filedialog
from PIL import ImageTk, Image
from tkinter import messagebox
import time
import soundfile as sf
import sounddevice as sd
import EBCTPS
import numpy as np
import matplotlib.animation as animation
import scipy.interpolate as si
from math import floor
import os

scaled_sp, x, Fs = 0, 0, 0
location = ''
scale = 0
perform_func = 0
T = 0
rec_a, rt1, rt2 = [], 0, 0
coord = {i:i for i in range(51) if i%2 == 0}
flag = 0
mic = 0

# Function for browsing the audio files                
def Browse():
    
    global root, location, lct, x, Fs, flag, duration, sampf
    
    location = lct.get() 
    root.filename = tkinter.filedialog.askopenfilename()
    location = root.filename        
    lct.delete(0, tk.END)
    lct.insert(tk.END, location)
    
    try:
        
        x, Fs = sf.read(location)
        flag = 0
        
        if len(x.shape) > 1:
            x = np.array([x[i][0] for i in range(len(x))])
        
        no = str(round(len(x)/Fs, 1))
        
        if len(no) != 4:
              
              if len(no) > 4:
                    no = no[: 4]
                    
              else:
                    no = '0'*(4-len(no)) + no
        
        duration.config(text = 'Duration: ' + no + ' s')
        sampf.config(text = 'Sample frequency : %5d Hz' %(Fs))
        
    except:
        messagebox.showerror('File Error', 'Choose a valid audio .wav file!')

# This sets the process to be time scaling
def setts():
    
    global perform_func, flag
    perform_func = 'ts'
    flag = 0

# This sets the process to be pitch scaling
def setps():
    
    global perform_func, flag
    perform_func = 'ps'
    flag = 0

#Function for plotting spectrograms for input and output audio signals
#No input 
def spec_plot():
      
      global x, scaled_sp, ax1, ax2, fig, cax1, Fs
      
      ax1.clear()
      ax2.clear()
      
      if type(x) == type(np.array([])):
            
            pxx1,  freq1, t1, cax1 = ax1.specgram(x = x, 
                         NFFT = 1024, 
                         Fs = Fs, 
                         noverlap = 512,
                         cmap = 'jet')            
            

            pxx2,  freq2, t2, cax2 = ax2.specgram(x = scaled_sp,
                         NFFT = 1024,
                         Fs = Fs,
                         noverlap = 512,
                         cmap = 'jet')
            
            cbar = fig.colorbar(cax2, ax3)
            cbar.set_label('Intensity (dB)')

      pad = -(len(x)/Fs/6.5)
      ax1.text(pad, 0, 'Frequency (Hz)', rotation = 'vertical')
      ax2.set_xlabel('Time (s)')    
      
      ax1.tick_params(
          axis='x',          
          which='both',      
          bottom='on',      
          top='off',         
          labelbottom='on')
          
      ax1.tick_params(
          axis='y',          
          which='both',     
          right='off',      
          left='on',        
          labelleft='on')
      
      ax2.tick_params(
          axis='x',          
          which='both',      
          bottom='on',      
          top='off',         
          labelbottom='on')
          
      ax2.tick_params(
          axis='y',          
          which='both',     
          right='off',      
          left='on',        
          labelleft='on')
      
      ax1.set_title('Input Spectrogram')
      ax2.set_title('Output Spectrogram')
      
      fig.tight_layout()
      
      
# The main function which fragments the audio and processes them and then combines them back by overlapping
def perform():
    
    global scale, perform_func, x, Fs, scaled_sp, flag, location, name
    
    if flag == 0:
          
          flag = 1
          
          try:
              
              discretisation = 1             # The audio is split into fragments of 1 second length
              T = len(x)/Fs                  # This is the total length of the audio in seconds
              ms = int(len(x)/(T/discretisation))
              subpart = [x[: ms]]            # subpart array stores the audio fragments  
              j = ms 
              val = int(len(x)/T*0.030)      
              hamwin = np.hanning(2*val)     # 60 milli seconds long hanning window made for overlapping
              zff_gci = EBCTPS.epoch(x, Fs)      
              zff = [zff_gci[: ms]]
              
              # The audio is split up
              while j+ms<len(x):
                  
                  subpart.append(x[j-val: j+ms])
                  zff.append(zff_gci[j-val: j+ms])
                  j += ms
                  
              if j < len(x)-1:
                  
                  subpart.append(x[j-val: ]) 
                  zff.append(zff_gci[j-val: ])
                  
              c1, c2 = dict(), dict()        # c1 stores the scale for each point of time extracted from the graph
                                             # c2 stores scale for a peiod of 1 second
              
              for i in range(len(points)):
          
                  scale = points[i][1]/points[i][0]
                  t = T*points[i][0]/50
                  c1[round(t, 2)] = round(scale, 3)
              
              b = list(c1.keys())[0]
              a = int(b)
              
              if b-a <= discretisation:
                  c = a+discretisation
              else:
                  c = a+1
              
              d = list(c1.keys())
              e = list(c1.values())
              copy = c
              
              for i in range(len(c1)):
                  
                  if i<len(c1)-1 and d[i] <= c <= d[i+1]:
                      c2[c] = (e[i+1]-e[i])/(d[i+1]-d[i])*(c-d[i]) + e[i]
                      c += 1
                  
                  if i+1 == len(c1):
                      c2[c] = c2[c-1]
                      
              va = list(c2.values())[0] 
              vb = list(c2.values())[-1]
              
              i=0
              while i<copy:
                  c2[i] = va
                  i += 1
                  
                  
              i=max(list(c2.keys()))
              while i<T:
                  c2[i] = vb
                  i += 1
                  
                  
              try:
                               
                scaled_sp = [] 
                    
                # EBCTPS() performs time scaling for each fragment audio
                if perform_func == 'ts':        
                    
                    w = 0
                    for i in range(len(subpart)):
                        
                        try:
                            scale = floor(list(c2.values())[list(c2.keys()).index(w)]*1000)/1000
                            w += discretisation
                        except:
                            scale = floor(list(c2.values())[list(c2.keys()).index(max(list(c2.keys())))]*1000)/1000
                            
                        subpart[i] = EBCTPS.ETS(subpart[i], Fs, scale, zff[i])
                        
                        if len(scaled_sp) == 0:
                            scaled_sp = np.array(subpart[i])
                        
                        else:
                            
                            # the audio fragments are 50% overlapped using the hanning window
                            temp1 = (scaled_sp[len(scaled_sp)-val: ]*hamwin[: val])
                            temp2 = (subpart[i][: val]*hamwin[val: ])
                            scaled_sp = np.concatenate((scaled_sp[: len(scaled_sp)-val], 
                                                                  temp1+temp2, 
                                                                  subpart[i][val: ]))
                
                # EBCTPS() performs time scaling for each fragment audio                                 
                elif perform_func == 'ps':
                    
                    w = 0
                    for i in range(len(subpart)):
                        
                        try:
                            scale = floor(list(c2.values())[list(c2.keys()).index(w)]*1000)/1000
                            w += discretisation
                        except:
                            scale = floor(list(c2.values())[list(c2.keys()).index(max(list(c2.keys())))]*1000)/1000
                            
                        subpart[i] = EBCTPS.EPS(subpart[i], Fs, scale, zff[i])
                        
                        if len(scaled_sp) == 0:
                            scaled_sp = list(subpart[i])
                        
                        else:
        
                            # the audio fragments are 50% overlapped using the hanning window
                            temp1 = (scaled_sp[len(scaled_sp)-val: ]*hamwin[val: ])
                            temp2 = (subpart[i][: val]*hamwin[: val])
                            scaled_sp = np.concatenate((scaled_sp[: len(scaled_sp)-val], 
                                                                  temp1+temp2, 
                                                                  subpart[i][val: ]))
            
                else:
                    messagebox.showerror('Error!', 'Choose time scaling or pitch scaling!')
                    
                sd.play(scaled_sp, Fs)   # the scaled audio is played
                
                spec_plot()              # plots the spectrograms on GUI
                              

                
                          
              except:
                    
                    messagebox.showerror('File Error!', 'Some error with the selected audio file!')
          
          except:
              
              messagebox.showerror('File Error!', 'Some error with the selected audio file!')

    else:
          sd.play(scaled_sp, Fs)
          
          
# Function for stopping the audio playing
def STop():
    
    sd.stop()

# Function to stop recording
def Stoprec():
      
    global rec_a, location, r2, rt1, x, Fs, flag, duration, flag, lct, mic
    
    if mic == 1:
          
          mic = 0
          flag = 0
          rt2 = time.time()
          sd.stop()
          
          n = 1
          
          while True:
                
                recname = 'Record_' + str(n) + '.wav'
                
                if recname in os.listdir(os.getcwd()):
                      n += 1
                      
                else:
                      sf.write(recname, rec_a[:int((rt2-rt1)/60*len(rec_a))], 16000)
                      location = recname
                      break
          
          lct.delete(0, tk.END)
          lct.insert(tk.END, location)
          
          try:
              
              x, Fs = sf.read(location)
              if len(x.shape) > 1:
                  x = np.array([x[i][0] for i in range(len(x))])
                
              no = str(round(len(x)/Fs, 1))
              
              if len(no) != 4:
                    
                    if len(no) > 4:
                          no = no[: 4]
                          
                    else:
                          no = '0'*(4-len(no)) + no
                                   
              duration.config(text = 'Duration: ' + no + ' s')
              sampf.config(text = 'Sample frequency : %5d Hz' %(Fs))
              
          except:
              messagebox.showerror('File Error!', 'Choose a valid audio .wav file!') 

    else:
          messagebox.showerror('Recording Fault!', 'Press Start before recording!')
                    
              

# Function to start recording       
def Startrec():
    
    global rec_a, r2, rt1, mic
    
    mic = 1
    
    rt1 = time.time()
    rec_a = sd.rec(frames=16000*60, 
                   samplerate=16000, 
                   channels=1, 
                   mapping=[2], 
                   blocking=False)

# Function to save scales audio 
# All saved audios will be in the Saved_audios Folder in the root folder of the script
def Save():

    global scaled_sp, savel, Fs, saveloc, name, location

    saveloc = savel.get()
  
    #Default saving of the processed audio in the root location of the real sample
    if saveloc == 'Save file name...' and type(scaled_sp) != type(0):
          
          n = 1
          
          while True:
               
                path = location[: -(location[:: -1].find('/') + 1)]
                
                try:
                      paths = os.listdir(path)
                      
                except:
                      paths = os.listdir(os.getcwd())
                      
                name = location[: -4] + '_' + perform_func + '_' + str(n) + '.wav'
                
                if name in [path + '/' + i for i in paths] or name in paths:
                      n += 1
                      
                else:                      
                      sf.write(name, scaled_sp, Fs)
                      break
    
    elif type(scaled_sp) != type(0):
      
          if '.wav' in saveloc:
              saveloc = saveloc[:-4]
          
          if flag == 1:
                os.remove(name)
                
          sf.write('Saved_audios/'+saveloc+'.wav', scaled_sp, Fs)
          
    else:
       messagebox.showerror('Error!', 'Some error has occurred. There is no audio to save!')   

# Function to get points for plotting the cubic B Spline
# pts is the array of the control point coordinates
# n is the number of points to be returned
# bspline() implemented from https://stackoverflow.com/questions/24612626/b-spline-interpolation-with-python
def bspline(pts, n=1000):
    
    pts = np.asarray(pts)
    l = len(pts)    
    ar = np.array([0, 0, 0] + list(range(l-2)) +[l-3]*3, dtype = 'int')
    t = np.linspace(1, l-3, n)
    arange = np.arange(len(t))
    points = np.zeros((len(t), pts.shape[1]))
    
    for i in range(pts.shape[1]):
        points[arange, i] = si.splev(t, (ar, pts[:, i], 3))

    return list(points)

# Function to change the control points on the graph dynamically
def animate(i):
    
    global coord, points
    
    ax.clear()
    points = bspline(list(zip(coord.keys(), coord.values())))
    
    for i in range(len(points)):
        points[i] = list(points[i])
        
    ax.plot([points[i][0] for i in range(len(points))],
            [points[i][1] for i in range(len(points))], 'g',
            list(coord.keys()), list(coord.values()), 'o--')
    
    ax.grid()
    ax.set_xlabel('Input Scale (%)')
    ax.set_ylabel('Output Scale (%)')
    
    labels = [str(i) for i in list(range(-20, 101, 20))]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    fig.tight_layout(pad = 2, h_pad = 2, w_pad = 2)
    

# Function that modifies the graph on mouse clicks   
def click(event):
    
    global x1, y1, coord, flag
    
    x1 = event.xdata
    y1 = event.ydata
    flag = 0
    
    if event.button == 1:
        
        if 0<= int(x1) <= 50 and x1 != None:
            
            if y1>50:
                y1 = 50
                
            elif y1<0:
                y1 =0
            
            coord[2*int(round(x1/2))] = y1
    
    elif event.button == 3:
        
        coord = {i:i for i in range(51) if i%2 == 0}
                  

''' <--- Code for the GUI layout starts here ---> '''
                  
root = tk.Tk()
root.wm_iconbitmap(r'favicon.ico')
root.title('Epoch Based Continuous Time and Pitch Scaling')
root.resizable(False, False)

#frame 0 for title image
frame0 = tk.Frame(root, borderwidth = 1, relief = tk.SUNKEN)
frame0.pack(side = tk.TOP, pady = 1)
img = ImageTk.PhotoImage(Image.open("title.png"))


panel = tk.Label(root, image = img, width = 1100)
panel.pack(side = tk.TOP)

# frame 1 for browsing files

frame1 = tk.Frame(root, borderwidth = 1, relief = tk.SUNKEN)
frame1.pack(side = tk.TOP)

select_file = tk.Label(frame1, text = 'Select WAV file:')
select_file.pack(side = tk.LEFT, padx = 10, pady = 5)

lct = tk.Entry(frame1, width = 92)
lct.insert(tk.END, 'Browse/Type file directory...')
lct.pack(side = tk.LEFT, padx = 10, pady = 5)

browse = tk.Button(frame1, text = 'Browse', command = Browse, width = 17)
browse.pack(side = tk.LEFT, padx = 15, pady = 5)

duration = tk.Label(frame1, text = 'Duration: 00.0 s')
duration.pack(side = tk.LEFT, padx = 10, pady = 5)

sampf = tk.Label(frame1, text = 'Sample frequency: 00000 Hz')
sampf.pack(side = tk.LEFT, padx = 10, pady = 5)

# frame 4 for recording an audio through mic
# frame 4 for selecting between the options time scaling and pitch scaling

frame4 = tk.Frame(root, borderwidth = 1, relief = tk.SUNKEN, width = 300)
frame4.pack(side = tk.TOP)

record = tk.Label(frame4, text = 'Record audio:')
record.pack(side = tk.LEFT, padx = 15, pady = 5)

start = tk.Button(frame4, text = 'Start', width = 26, command = Startrec)
stop = tk.Button(frame4, text = 'Stop', width = 26, command = Stoprec)

start.pack(side = tk.LEFT, padx = 15, pady = 5)
stop.pack(side = tk.LEFT, padx = 15, pady = 5)


bar1 = tk.Label(frame4, text = '||', font = ("Arial", 20), fg = 'Gray')
bar1.pack(side = tk.LEFT)

select_perform = tk.Label(frame4, text = 'Perform:')
select_perform.pack(side = tk.LEFT, padx = 29, pady = 5)

rb1 = tk.Radiobutton(frame4, text = 'Time scaling', 
                     value = 1, indicatoron = 0, 
                     width = 26, command = setts)

rb2 = tk.Radiobutton(frame4, text = 'Pitch Scaling', 
                     value = 2, indicatoron = 0, 
                     width = 26, command = setps)

rb1.pack(side = tk.LEFT, padx = 15, pady = 5)
rb2.pack(side = tk.LEFT, padx = 15, pady = 5)

# frame 6 for playing the scaled audio
# frame 6 for saving the scaled audio file into the Saved_audios folder

frame6 = tk.Frame(root, borderwidth = 1, relief = tk.SUNKEN)
frame6.pack(side = tk.TOP)

space1 = tk.Label(frame6, text = '                                 ')
space1.pack(side = tk.LEFT)

perform = tk.Button(frame6, text = 'Play', width = 26, command = perform)
perform.pack(side = tk.LEFT, padx = 16, pady = 5)

stop1 = tk.Button(frame6, text = 'Stop', width = 26, command = STop)
stop1.pack(side = tk.LEFT, padx = 15, pady = 5)


bar2 = tk.Label(frame6, text = '||', font = ("Arial", 20), fg = 'Gray')
bar2.pack(side = tk.LEFT)

space2 = tk.Label(frame6, text = '      ')
space2.pack(side = tk.LEFT)

savel = tk.Entry(frame6, width = 49)
savel.pack(side = tk.LEFT, padx = 5)
savel.insert(tk.END, 'Save file name...')

save = tk.Button(frame6, text = 'Save scaled audio', 
                 command = Save, width = 26)

save.pack(side = tk.LEFT, padx = 15, pady = 5)

# frame 3 for plotting the graph of cubic B Spline and spectrograms

frame3 = tk.Frame(root, borderwidth = 1, relief = tk.SUNKEN)
frame3.pack(side = tk.TOP)

gs = mpl.gridspec.GridSpec(2, 17)

fig = Figure(figsize = (10, 4), dpi = 110)
ax = fig.add_subplot(gs[:, :8])

ax1 = fig.add_subplot(gs[0, 8: 16])
ax1.set_title('Input Spectrogram')
ax1.plot()

ax2 = fig.add_subplot(gs[1, 8:16])
ax2.set_title('Output Spectrogram')
ax2.plot()

ax3 = fig.add_subplot(gs[:, 16])
ax3.plot()

cb = mpl.colorbar.ColorbarBase(ax3, orientation='vertical')
cb.set_label('Intensity (dB)')

ax1.tick_params(
    axis='x',          
    which='both',      
    bottom='off',      
    top='off',         
    labelbottom='off')
    
ax1.tick_params(
    axis='y',          
    which='both',     
    right='off',      
    left='off',        
    labelleft='off')

ax2.tick_params(
    axis='x',          
    which='both',      
    bottom='off',      
    top='off',         
    labelbottom='off')
    
ax2.tick_params(
    axis='y',          
    which='both',     
    right='off',      
    left='off',        
    labelleft='off')
    
canvas = FigureCanvasTkAgg(fig, root)
canvas.draw()
canvas.get_tk_widget().pack()

ani  = animation.FuncAnimation(fig, animate, interval = 100)

b1 = fig.canvas.mpl_connect('button_press_event', click)

# Frame 7 for details
frame7 = tk.Frame(root, borderwidth = 1, relief = tk.SUNKEN)
frame7.pack(side = tk.TOP)

details = tk.Label(frame7, anchor = tk.W, width = 160, 
                   text = '   GUI for Epoch Based Continuous Time and Pitch Scaling algorithm developed in Spectrum Lab, Electrical Engineering department, Indian Instititute of Science, Bangalore developed using Python 3.6')

details.pack(side = tk.LEFT)

# Function  for destroying the windows
def on_closing():
    sd.stop()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()