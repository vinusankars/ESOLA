# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:09:32 2017

Coded by: S Vinu Sankar

This python file, GUI.py contains all the code for the GUI.

This program would get an input scale from the user.
And the input audio file will be scaled according to the input scale

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
import matplotlib.animation as animation
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
import os

scaled_sp, x, Fs = 0, 0, 0
location = ''
scale = 0
perform_func = 0
T = 0
rec_a, rt1, rt2 = [], 0, 0
flag, f = 0, 1
mic = 0
zff = 0
t1, t2 = 0, 0

# Function for browsing the audio files                
def Browse():
    
    global root, location, lct, x, Fs, flag, duration, sampf, zff
    
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
                    
        zff = EBCTPS.epoch(x, Fs)
        
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

# Function for plotting spectrograms for input and output audio signals
# No input 
def spec_plot():
      
      global x, scaled_sp, ax1, ax2, ax3, fig, Fs
      
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
            cbar.set_width = '1'

      pad = -(len(x)/Fs/6.5)*0.8
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

# The function that plays the processed audio file
# It considers playing the processed file dynamically/continuously as the scale gets changed
# scaled_sp is the processed signal
# Fs is its sample rate
# play(scaled_sp, Fs) plays the audio scaled_sp at rate Fs
def play(scaled_sp, Fs):
     
    global t1, t2
    
    sd.stop()
    
    if t2 != 0 and t2-t1-1 >= 0:
          
          cont = int((t2-t1)/(len(scaled_sp)/Fs)*len(scaled_sp))%len(scaled_sp)
          sd.play(scaled_sp[cont: ], Fs)
          t2 = 0
    
    else:        
          sd.play(scaled_sp, Fs)           
      
# The main function which fragments the audio and processes them and then combines them back by overlapping
def perform():
    
    global scale, perform_func, x, Fs, scaled_sp, flag, location, name, zff, sscale, t1, t2, f
    
    scale = sscale.get()
    t1, t2 = 0, 0
    f = 0
    
    if flag == 0:
          
          flag = 1
          
          if perform_func == 'ts':
                
                scaled_sp = EBCTPS.ETS(x, Fs, scale, zff)
      
          elif perform_func == 'ps':
                
                scaled_sp = EBCTPS.EPS(x, Fs, scale, zff)
                
          else:
                messagebox.showerror('Error!', 'Choose either time scaling or pitch scaling!')
                
          spec_plot()
          t1 = time.time()
          play(scaled_sp, Fs)

    else:
          t1 = time.time()
          play(scaled_sp, Fs)
          
          
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

# Just a pass function for animating graphs
def animate(i):      
      pass

# This function takes care of the scaling bar sscale
# This function is performed when the scale bar is modified
def SCale(var):
    
    global sscale, scale, scaled_sp, x, Fs, zff, flag, t1, t2, f
    
    scale = sscale.get()    
    t2 = time.time()
    
    flag = 0

    if perform_func == 'ps':          
      
          if f == 0:
                scaled_sp = EBCTPS.EPS(x, Fs, scale, zff)
                
          play(scaled_sp, Fs)
          spec_plot()
      
    elif perform_func == 'ts':
      
          if f == 0:
                scaled_sp = EBCTPS.ETS(x, Fs, scale, zff) 
                
          play(scaled_sp, Fs)
          spec_plot()



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

lct = tk.Entry(frame1, width = 40)
lct.insert(tk.END, 'Browse/Type file directory...')
lct.pack(side = tk.LEFT, padx = 10, pady = 5)

browse = tk.Button(frame1, text = 'Browse', command = Browse, width = 20)
browse.pack(side = tk.LEFT, padx = 15, pady = 5)

sscale = tk.Scale(frame1, command = SCale, from_ = 0.5, to = 2.0, length = 255,
                  sliderlength = 20, orient = tk.HORIZONTAL, resolution = 0.1)

sscale.pack(side = tk.LEFT, padx = 15, pady = 0)
sscale.set(0.5)
scale = sscale.get()

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
# frame 6 for saving the scaled audio file into the Records folder

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

# frame 3 for plotting the spectrograms of output and input signals

frame3 = tk.Frame(root, borderwidth = 1, relief = tk.SUNKEN)
frame3.pack(side = tk.TOP)


fig = Figure(figsize = (10, 4), dpi = 110)

gs = mpl.gridspec.GridSpec(2, 17)

ax1 = fig.add_subplot(gs[0, 1:13])
ax1.set_title('Input Spectrogram')
ax1.plot()

ax2 = fig.add_subplot(gs[1, 1:13])
ax2.set_title('Output Spectrogram')
ax2.plot()

ax3 = fig.add_subplot(gs[: , 14])
ax3.plot()

ax4 = fig.add_subplot(gs[: , 0])
ax4.plot()
ax4.axis('off')

ax5 = fig.add_subplot(gs[: , 16])
ax5.plot()
ax5.axis('off')

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

cb = mpl.colorbar.ColorbarBase(ax3, orientation='vertical')
cb.set_label('Intensity (dB)')
    
canvas = FigureCanvasTkAgg(fig, root)
canvas.draw()
canvas.get_tk_widget().pack()

ani  = animation.FuncAnimation(fig, animate, interval = 100)
fig.tight_layout()

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