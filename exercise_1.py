import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

pi = np.pi


# All discrete values of t in the interval from t0 and t1
def true_func(func):
    func = np.vectorize(func)
    t = np.linspace(t_min, t_max, Ts)
    f_true = func(t)                    # the true f(t)
    return f_true

# Function that computes the real fourier couples of coefficients (a0, 0), (a1, b1)...(aN, bN)
def compute_real_fourier_coeffs(func, N):
    an = []
    bn = []
    for n in range(N+1):
        a = (2./T) * spi.quad(lambda t: func(t) * np.cos(2*pi*n*t/T), 0, T)[0]
        b = (2./T) * spi.quad(lambda t: func(t) * np.sin(2*pi*n*t/T), 0, T)[0]
        an.append(a)
        bn.append(b)
    return an, bn

# Function that computes the real form Fourier series using an and bn coefficients F_N(t)
def fit_func_by_fourier_series_with_real_coeffs(f, t, an, bn, N):
    [an, bn] =  compute_real_fourier_coeffs(f, N)
    fn = 0.
    for n in range(0, len(an)): 
        if n > 0:
            fn +=  an[n] * np.cos(2*pi*n*t/T) + bn[n]*np.sin(2*pi*n*t/T)
        else:
            fn +=  an[0]/2
    return fn

#function to integrate on complex field
def complex_quad(func, a, b, **kwargs):
    def real_func(t):
        return np.real(func(t))
    def imag_func(t):
        return np.imag(func(t))
    real_integral = spi.quad(real_func, a, b, **kwargs)
    imag_integral = spi.quad(imag_func, a, b, **kwargs)
    integral = (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])
    return integral

# Function that computes the complex fourier coefficients c-N,.., c0, .., cN
def compute_complex_fourier_coeffs(func, N):
    cn = []
    for n in range(-N, N+1):
        c = (1./T) * complex_quad(lambda t: func(t) * np.exp(-1j*2*pi*n*t/T), 0, T)[0]
        cn.append(c)
    return np.array(cn)

# Function that computes the complex form Fourier series using cn coefficients G_N(t)
def fit_func_by_fourier_series_with_complex_coeffs(t, cn, N):
    gn = 0. + 0.j
    for n in range(-N, N+1):
        c = cn[n+N]
        gn +=  c*np.exp(1j*2.*pi*n*t/T)
    return gn

# Visualize results
def plot_graph(t, f_original, fn, gn, N):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    fig.suptitle('REAL FUNCTION WITH N = ' + str(N) )
    # plt.sca(axes[0])
    # plt.plot(t, f_original, color='black', linewidth=1.25, label='f(t)')
    # plt.title('f(t)')
        
    plt.sca(axes[0])
    plt.plot(t, f_original, color='black', linewidth=1.25)
    plt.plot(t, fn, color='green', linewidth=1)
    plt.title('F_N')
    plt.legend(['f(t)', 'F_N(t)'], loc='upper right')
        
    plt.sca(axes[1])
    plt.plot(t, f_original, color='black', linewidth=1.25)
    plt.plot(t, gn, color='red', linewidth=1)
    plt.title('G_N')
    plt.legend(['f(t)', 'G_N(t)'], loc='upper right')
    
    for i in range(2):
        plt.sca(axes[i])
        plt.grid(color = 'silver', linestyle = '--', linewidth = 0.5)
        
    plt.show()

# Parseval check
def parseval(an, bn, cn, N, f):
    t = np.linspace(-pi, pi, 10000)
    abs_f = np.vectorize(lambda t: abs(f(t)))
    dx = t[1] - t[0]
    func_norm = np.dot(abs_f(t), abs_f(t)) * dx
    f_ab = pi*(an[0]**2/2 + sum([an[i]**2 + bn[i]**2 for i in range(1, N+1)]))
    f_c = 2*pi*sum(abs(cn[i])**2 for i in range(len(cn)))
    return func_norm, f_ab, f_c


T = 2*pi            # period value
t_min = -2*pi       # initian value of t (begin time)
t_max = 2*pi        # final value of t (end time)
Ts = 1000           # number of discrete values of t between t0 and t1
t_range = np.linspace(t_min, t_max, Ts)
N_s = np.array([1, 2, 10, 20, 50])

## ======================= The periodic real-valued function f(t) with period equal to T =========================
# f_1 = lambda t: a if t0 <= t % T < t1 else b
# f_2 = lambda t: abs(np.sin(t)) + np.cos(t)
# f_3 = lambda t: np.sin(t)*np.cos(2*t)
# f_4 = lambda t: np.cos(2*t)*np.sin(3*t)


## ========================== The true function ================================
# f_1_true = true_func(f_1)
# f_2_true = true_func(f_2)
# f_3_true = true_func(f_3)
# f_4_true = true_func(f_4)



## ========================== Task 1: [an, bn, cn] ==============================
# Initial Parameters
# a = 2; b = 3
# t0 = 1; t1 = 2; t2 = 4
# T = t2 - t0
# N = 50

# # The periodic real-valued function f(t) with period equal to T
# f_1 = lambda t: a if t0 <= (t % T) < t1 else b
# # The true function
# f_1_true = true_func(f_1)


# for a in an:
#     print(a)
# print('\n')
# for b in bn:
#     print(b)
# print('\n')
# for i in range(len(cn)):
#     print(cn[i])


# for N in N_s:
#     [an, bn] = compute_real_fourier_coeffs(f_4, N)
#     fn = fit_func_by_fourier_series_with_real_coeffs(f_4, t_range, an ,bn, N)
#     cn = compute_complex_fourier_coeffs(f_4, N)
#     gn = fit_func_by_fourier_series_with_complex_coeffs(t_range, cn, N)
#     plot_graph(t_range, f_4_true, fn, gn, N)
    
    
    
    
