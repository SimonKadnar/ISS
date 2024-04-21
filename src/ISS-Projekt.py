import numpy as np
from scipy.io import wavfile 
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
import scipy.signal as signal
import math
import cmath

def normaliz(data): #pomocna funckia pre normlaizovanie hodnot
    average = 0
    for i in data:
        average += i
    average = average/data.size
    data = data-average
    data = data/max(abs(data))
    return data

def uloha1(fs,data):
    print("Úloha 1:")
    print("pocet vzorkov:",data.size)
    print("dlzka v sekundach:",data.size/fs)
    print("minimalna hodnota vzorku:",min(data))
    print("maximalna hodnota vzorku:" ,max(data))

    t = np.arange(data.size) / fs
    plt.figure(figsize=(6,3))
    plt.plot(t, data)
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Úloha 1') #frekvence v hercoch
    plt.gca().set_ylabel('Hodnota vzorku')
    plt.tight_layout()
    plt.show()

def uloha2(fs,data):
    
    data = normaliz(data)           #normalizacia...

    pole = np.array([])
    matica = np.array([])
    i = 0
    cnt = 0
    while i != len(data):           #delenie do ramcov
        if cnt < 1023:
            pole = np.insert(pole,cnt,data[i])
            cnt = cnt+1
            i = i+1
        elif cnt == 1023:
            pole = np.insert(pole,cnt,data[i])
            matica = np.append(matica,pole)
            pole = np.array([])
            cnt = 0
            i = i-511

    matica = np.reshape(matica,(-1,1024))   #preskladanie pola na maticu

    i = 11  #najkrajsi ramec ktory sa mi podarilo najst
    data = matica[i]
    t = np.arange(data.size)/fs
    plt.figure(figsize=(6,3))
    plt.plot(t, data)
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Úloha 2')
    plt.tight_layout()
    plt.show()

def uloha4(fs,data):
    
    data = normaliz(data)       #normalizacia

    matica = np.array([])
    nic_1,nic_2,matica = spectrogram(data,fs=8000,nperseg=1024,noverlap=512)    #vykonový spektogram
    matica = 10* np.log10(matica)   
    
    plt.figure(figsize=(9,3))
    plt.imshow(matica,origin="lower",aspect="auto",extent=[0,data.size/fs,0,fs/2])
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvencia[Hz]')
    plt.gca().set_title('Úloha 4')
    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

def uloha5(fs,data):

    print("Úloha 5:")
    print ("Prvá rušívá komponenta má frekenciu : 950")
    print ("Druhá rušívá komponenta má frekenciu : 1900")
    print ("Tretia rušívá komponenta má frekenciu : 2850")
    print ("Štrvtá rušívá komponenta má frekenciu : 3780")
    #950
    #1900
    #2850
    #3800

def uloha6(fs,data):

    frekvencie = (950,1900,2850,3780)   #frekvencie rusivych komponent
    
    pole = np.array([])
    for i in range(data.size):      #vytvorenie pole o dlzke casu
        pole = np.append(pole,i/fs)

    out1 = np.cos(2 * np.pi * frekvencie[0] * pole) 
    out2 = np.cos(2 * np.pi * frekvencie[1] * pole)
    out3 = np.cos(2 * np.pi * frekvencie[2] * pole)
    out4 = np.cos(2 * np.pi * frekvencie[3] * pole)
    
    out = np.array([])
    out = out1 + out2 + out3 + out4 #spojenie vsetkych rusivych komponent do jednej

    out = normaliz(out)
    
    wavfile.write('../audio/4cos.wav',fs,(out*np.iinfo(np.int16).max).astype(np.int16))

    nic_1,nic_2,out_final = spectrogram(out,fs,nperseg=1024)
    out_final = 10* np.log10(np.abs(out_final))
    
    plt.figure(figsize=(9,3))
    plt.imshow(out_final,origin="lower",aspect="auto",extent=[0,out.size/fs,0,fs/2])
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvencia[Hz]')
    plt.gca().set_title('Úloha 6')
    cbar = plt.colorbar()
    cbar.set_label('Spektralna hustota výkonu[dB]', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

def uloha7(fs,data):

    n = np.array([])

    w = 2 * np.pi * 950/ fs
    n = np.append (n,math.e**(w*1j))
    w = 2 * np.pi * 1900/ fs
    n = np.append (n,math.e**(w*1j))
    w = 2 * np.pi * 2850/ fs
    n = np.append (n,math.e**(w*1j))
    w = 2 * np.pi * 3780/ fs
    n = np.append (n,math.e**(w*1j))

    pole = np.array([])
    pole = np.conjugate(n)      #komplexne zdruzene body

    filter = np.array([])
    filter = np.append(filter,n)
    filter = np.append(filter,pole)

    filter = np.poly(filter)    

    Nimp = 9
    imp = [1, *np.zeros(Nimp-1)] # jednotkovy impuls
    imp = lfilter(filter, [1], imp)

    plt.figure(figsize=(5,3))
    plt.stem(np.arange(Nimp), imp, basefmt=' ')
    plt.gca().set_xlabel('$n$')
    plt.gca().set_title('Úloha 7 Impulzná odozva $h[n]$')
    plt.grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.show()
        
    return filter

def uloha9(fs,data,filter):

    w, H = freqz(filter,[1])

    plt.figure(figsize=(6,3))
    plt.plot(w / 2 / np.pi * fs, np.abs(H))
    
    plt.gca().set_xlabel('Frekvencia[Hz]')
    plt.gca().set_title('Úloha 9 Modul frekvenčnej charakteristiky $|H(e^{j\omega})|$')
    plt.grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.show()
    

def uloha10(fs,data,filter):
    
    data = normaliz(data)       #normalizacia

    out = lfilter(filter, [1], data) #pouzitie filtra na signal

    out = normaliz(out)

    t = np.arange(out.size) / fs
    plt.figure(figsize=(6,3))
    plt.plot(t, out)
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Úloha 10')
    plt.tight_layout()
    plt.show()

    wavfile.write('../audio/clean_z.wav',fs,(out*np.iinfo(np.int16).max).astype(np.int16))  #generovanie signalu
    
fs, data = wavfile.read('../audio/xkadna00.wav')    #nacitanie vstupneho signalu

uloha1(fs,data)
uloha2(fs,data)

uloha4(fs,data)
uloha5(fs,data)
uloha6(fs,data)
filter = uloha7(fs,data)

uloha9(fs,data,filter)
uloha10(fs,data,filter)
    












