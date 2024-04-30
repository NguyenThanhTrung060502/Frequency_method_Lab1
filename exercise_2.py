import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

pi = np.pi

# Function f(t)
def f(t):
    t = (t + T/8) % T - T/8
    real_part = -1
    imag_part = -1
    if -T/8 <= t < T/8:
        real_part = R
        imag_part = 8*R*t/T
    if T/8 <= t < 3*T/8:
        real_part = 2*R - 8*R*t/T 
        imag_part = R
    if 3*T / 8 <= t < 5*T/8:
        real_part = -R
        imag_part = 4*R - 8*R*t/T
    if 5*T/8 <= t <= 7 * T/8:
        real_part = -6*R + 8*R*t/T
        imag_part = -R

    return real_part + 1j*imag_part

# Coordinate axis Oxy
def Oxy(axes):
    plt.sca(axes)
    Ox = list(range(-10, 11, 1))
    Oy = list(range(-5, 6, 1))
    plt.xticks(Ox, color='black')
    plt.yticks(Oy, color='black')

# Function to integrate on complex field
def complex_quad(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = spi.quad(real_func, a, b, **kwargs)
    imag_integral = spi.quad(imag_func, a, b, **kwargs)
    integral = (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])
    return integral

# Function that computes the complex fourier coefficients c-N,.., c0, .., cN
def compute_complex_fourier_coeffs(func, N):
    cn = []
    for n in range(-N, N+1):
        c = (1./T)*complex_quad(lambda t: func(t)*np.exp(-1j*2*pi*n*t/T), 0, T)[0]
        cn.append(c)
    return np.array(cn)

# Function that computes the complex form Fourier series using cn coefficients
def fit_func_by_fourier_series_with_complex_coeffs(t, cn):
    gn = 0. + 0.j
    for n in range(-N, N+1):
        c = cn[n+N]
        gn +=  c * np.exp(1j*2*pi*n*t/T)
    return gn

# Visualize the results
def plot_graph(t, f_true_real, f_true_imag, gn):
    gn_real = [y.real for y in gn]
    gn_imag = [y.imag for y in gn]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    fig.suptitle('COMPLEX FUNCTION, WITH CASE N = ' + str(N))
    for i in range(2):
        for j in range(2):
            Oxy(axes[i][j])
            if i == 0 and j == 0:
                plt.sca(axes[i][j])
                plt.plot(f_true_real, f_true_imag, color='seagreen', linewidth=1.25, label='f(t)')
                plt.xlabel('Re')
                plt.ylabel('Im')
                plt.grid('True')
                plt.title('Original function')
            if i == 0 and j == 1 :
                plt.sca(axes[i][j])
                plt.plot(f_true_real, f_true_imag, color='black', linewidth=1.25, label='f(t)')
                plt.plot(gn_real, gn_imag, color='red', linewidth=1.25, label='G_N')
                plt.title('Fourier funtion ' )
            if i == 1 and j == 0:
                plt.sca(axes[i][j])
                plt.plot(t, f_true_real, color='black', linewidth=1.25, label='Original')
                plt.plot(t, gn_real,     color='red',   linewidth=1, label='Fourier')
                plt.title('Real part', loc='left')
            if i == 1 and j == 1:
                plt.sca(axes[i][j])   
                plt.plot(t, f_true_imag, color='black', linewidth=1.25, label='Original')
                plt.plot(t, gn_imag,     color='red',   linewidth=1, label='Fourier')
                plt.title('Imag part', loc='left')
            plt.grid(color = 'silver', linestyle = '--', linewidth = 0.5)
            plt.legend()
    plt.show()

# Parseval check
def parseval(f):
    cn = compute_complex_fourier_coeffs(f, N)
    t = np.linspace(-pi, pi, 10000)
    abs_f = np.vectorize(lambda t: abs(f(t)))
    dx = t[1] - t[0]
    func_norm = np.dot(abs_f(t), abs_f(t))*dx
    f_c = 2*pi*sum(abs(cn[i])**2 for i in range(len(cn)))
    return func_norm, f_c
    
T = 4           
R = 3
t0 = -T/8        
t1 = 7*T/8         
Ts = 1000       
N_s = np.array([1, 2, 3, 10])

f = np.vectorize(f)
t_range = np.linspace(t0, t1, Ts)
f_true = f(t_range)                         
f_true_real = [y.real for y in f_true]
f_true_imag = [y.imag for y in f_true]


for N in N_s:
    cn = compute_complex_fourier_coeffs(f, N)
    gn = fit_func_by_fourier_series_with_complex_coeffs(t_range, cn)
    plot_graph(t_range, f_true_real, f_true_imag, gn)
    print(parseval(f))
