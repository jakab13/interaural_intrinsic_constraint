import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


df = pd.read_csv('C:/Users/km49dola/Desktop/ma/recordings/kemar/kemar99/kemar99_2025-03-28-09-47-51.csv')
subject= 'kemar81'
unique_freqs = df.iloc[:, 1].unique()
unique_freqs = df['freq'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_freqs)))

#itd fitting
#itd=r(w+sin(w))/C -> woodworth equation only usable between -90° and 90°

def woodworth(x, r):
    w = ((np.pi)/180)*x  #angles in radians
    return  r/100 * (w + (np.sin(w)))/343

plt.figure(figsize=(10, 6))
r_itd={}
for i, freq in enumerate(unique_freqs):
    df_filtered = df[df.iloc[:, 1] == freq]
    azimuth = df_filtered.iloc[:, 0].values
    itd = df_filtered.iloc[:, 11].values

    params, _ = curve_fit(woodworth, azimuth, itd, p0=[0.1])#, p0=[1, max(ild),0.2,np.median(azimuth), 0]
    r_opt = float(params[0])
    r_itd[float(freq)]=r_opt
    #print(f"for {freq} Hz:  r= {r_opt:.5f}")
    azimuth_fit = np.linspace(min(azimuth), max(azimuth), 100)
    itd_fit = woodworth(azimuth_fit, *params)
    plt.scatter(azimuth, itd, color=colors[i], label=f"{freq} Hz", alpha=0.6)
    plt.plot(azimuth_fit, itd_fit, color=colors[i], linestyle="dashed")

plt.xlabel('azimuth')
plt.ylabel('itd')
plt.title('curve fit itd')
plt.legend(title='frequency (Hz)')
plt.savefig('C:/Users/km49dola/Desktop/ma/recordings/kemar/kemar99/curve_fit_kemar99_itd')
plt.show()
print(r_itd)

#ild fitting


def ild_func(x, g, c):
    x= np.pi/180 *x
    r1= 1 - g * np.cos(x + (np.pi / 2)) # radian, asymmetry
    r2 = 1 - (1-g)* np.cos(x - (np.pi/2))
    k = 2 * np.pi * (freq / 343) # wavenumber
    y1 = 1/ (1+(k*0.09*r1)**2)
    y2 = 1/ (1+(k*0.09*r2)**2)
    f = c * (20 * np.log10(y2) - 20 * np.log10(y1))

    return f
'''
def ild_func(x, a_right, a_left):
    r = ((np.pi) / 180) * x
    gr = 1 - a_right * np.cos(r - 90)
   # gr = 1 - a * np.cos(r - 90)
   # gl= 1 - (1 - a) * np.cos(r+90)
    gl= 1 - a_left * np.cos(r+90)
    k=2*np.pi*freq/343
    hl=(1/(1+(k*gl)**2))
    hr=(1/(1+(k*gr)**2))
    f =  (20*np.log10(hl/(2*(10**-5)))) - (20*np.log10(hr/(2*(10**-5))))

    return f
'''
g_ild={}
c_ild={}
for i, freq in enumerate(unique_freqs):
    df_filtered = df[df.iloc[:, 1] == freq]
    azimuth = df_filtered.iloc[:, 0].values
    ild = df_filtered.iloc[:, 10].values

    params, _ = curve_fit(ild_func, azimuth, ild)
    g_opt, c_opt= params
    g_ild[float(freq)]=g_opt
    c_ild[float(freq)]=c_opt
    #print(f"{freq} Hz:  g= {g_opt}, c={c_opt}")
    azimuth_fit = np.linspace(min(azimuth), max(azimuth), 100)
    ild_fit = ild_func(azimuth_fit, *params)
    plt.scatter(azimuth, ild, color=colors[i], label=f"{freq} Hz", alpha=0.6)
    plt.plot(azimuth_fit, ild_fit, color=colors[i], linestyle="dashed")

plt.xlabel('azimuth')
plt.ylabel('ild')
plt.title('curve fit ild')
plt.legend(title='frequency (Hz)')
plt.savefig('C:/Users/km49dola/Desktop/ma/recordings/kemar/kemar99/curve_fit_kemar99_ild')
plt.show()
print(g_ild)
print(c_ild)

'''
#csv table
# rows: participants, columns: r_itd_600hz, g_ild_600,800,100,1200, c_ild_600,800,1000,1200
r_itd[600]

parameters={
    'subject': subject,
    'r_itd': r_itd,
    'g_ild': g_ild,
    'c_ild': c_ild
}
pd.DataFrame(data=parameters)
'''