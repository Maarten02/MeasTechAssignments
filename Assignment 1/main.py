import numpy as np
import matplotlib.pyplot as plt
import math

E1 = True
E2 = True
E3 = True
E4 = True
E5 = True


def cor(arr1, arr2):
    return np.sum((arr1 - np.average(arr1)) * (arr2 - np.average(arr2))) / len(arr1)

def get_Rxx(sig, ave_sig, shift):
    N_overlap = len(sig) - shift
    product = (sig[:N_overlap] - ave_sig) * (sig[shift:] - ave_sig)
    Rxx = np.sum(product) / N_overlap

    return Rxx

def get_Cxx(max_i, signal):
    ave_sig = np.average(signal)
    Rxx_zero = get_Rxx(signal, ave_sig, 0)
    autocorfunc = [1]
    for shift in range(1, max_i+1):
        Cxx = get_Rxx(signal, ave_sig, shift) / Rxx_zero
        autocorfunc.append(Cxx)

    return range(max_i+1), autocorfunc

Sx = lambda signal: np.sqrt(1/(len(signal)-1) * np.sum((signal - np.average(signal)) ** 2))

if E1:
    ###############
    ## Problem 1 ##
    ###############
    print('\n', 15*'=', '\n== Problem 1 == \n', 15*'=', sep='')
    # 1a) dependent: p, independent: rho

    # 1b)
    T_ave = 286
    R = 287.06
    p_arr = np.linspace(50000, 150000, 401)
    T_arr = 281+10 * (p_arr-50000)/100000
    rho_arr = p_arr / (R * T_arr)

    # 1c)
    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(p_arr, rho_arr, label='unshuffled meas.')
    #fig.savefig('figures/p_vs_rho.pdf')

    # 1d)
    p_vs_rho_linfit = np.polyfit(p_arr, rho_arr, 1)
    print(f'Linear fit rho = {p_vs_rho_linfit[0]:.6g}p + {p_vs_rho_linfit[1]:.4g}')
    rho_fit = p_arr * p_vs_rho_linfit[0] + p_vs_rho_linfit[1]
    ax.plot(p_arr, rho_fit, label='Linear fit [N-SHUF]')

    # 1e)
    standard_error = np.sqrt(np.sum((rho_arr - rho_fit) ** 2) / (401 - (1 + 1)))
    print(f'Standard error = {standard_error:.3g}')

    # 1f) INCORRECT!
    confidence_uppend = 2 * standard_error * np.sqrt(1/401 + (150000 - 100000) / np.sum((p_arr - 100000)**2))
    print(f'95% confidence on upper end of interval  = (+/-) {confidence_uppend:.3g}')

    # 1g)
    shuffled_indices = np.random.permutation(len(T_arr))
    shuffled_T = T_arr[shuffled_indices]
    rho_arr_sh = p_arr / (R * shuffled_T)

    # 1h) intercept INCORRECT!
    p_vs_rho_linfit_sh = np.polyfit(p_arr, rho_arr_sh, 1)
    print(f'\n[SHUFFLED] Linear fit rho = {p_vs_rho_linfit_sh[0]:.6g}p + {p_vs_rho_linfit_sh[1]:.4g}')
    rho_fit_sh = p_arr * p_vs_rho_linfit_sh[0] + p_vs_rho_linfit_sh[1]
    ax.plot(p_arr, rho_fit_sh, label='Linear fit [SHUF]')

    # 1i)
    standard_error_sh = np.sqrt(np.sum((rho_arr_sh - rho_fit_sh) ** 2) / (401 - (1 + 1)))
    print(f'[SHUFFLED] Standard error = {standard_error_sh:.3g}')

    # 1j)
    confidence_uppend_sh = 2 * standard_error_sh * np.sqrt(1/401 + (150000 - 100000) / np.sum((p_arr - 100000)**2))
    print(f'[SHUFFLED] 95% confidence on upper end of interval  = (+/-) {confidence_uppend_sh:.3g}')

    # plot figure
    plt.legend()
    plt.show()

    # Ideal gas law with average temperature
    rho_ideal_gas_T_ave = p_arr / (R * T_ave)
    true_A = 1 / (R*T_ave)

    # 1k) error A and Systematic error B
    print(f'\n[N-SHUFFLED] error A = {p_vs_rho_linfit[0] - true_A:.3g} error B = {p_vs_rho_linfit[1]:.3g}')
    print(f'[SHUFFLED] error A = {p_vs_rho_linfit_sh[0] - true_A:.3g} error B = {p_vs_rho_linfit_sh[1]:.3g}')

    # 1l) because the random spread is larger when shuffling

if E2:
    ###############
    ## Problem 2 ##
    ###############
    print('\n', 15 * '=', '\n== Problem 2 == \n', 15 * '=', sep='')
    system_resp = np.genfromtxt('data/system_response_2024.dat', skip_header=1)
    time = system_resp[:, 0]
    recorded_position = system_resp[:, 1]

    A = 2.2 # [cm^2]
    k = 1.8e3 # [N/m]

    # 1a) Determine the magnitude of the stepwise change in pressure (i.e. step height).
    magnitude = recorded_position[-1] - recorded_position[0]
    print(f'Step change = {magnitude:.3g} [mm]')

    # 1b) Determine the natural frequency and the damping ratio.
    max1idx = np.argmax(recorded_position)
    min1idx = np.argmin(recorded_position[max1idx:]) + max1idx
    max2idx = np.argmax(recorded_position[min1idx:]) + min1idx

    max1 = recorded_position[max1idx]
    max2 = recorded_position[max2idx]

    Td = time[max2idx] - time[max1idx]
    damping_ratio = 1 / (np.sqrt(1 + ((2 * np.pi) / (np.log((max1-recorded_position[-1]) / (max2-recorded_position[-1])))) ** 2))
    omega_d = 2*np.pi / Td
    omega_n = omega_d / np.sqrt(1 - damping_ratio ** 2)
    T_n = 2 * np.pi / omega_n

    print(f'natural frequency = {1/T_n:.3g} [Hz]') # incorrect
    print(f'first peak = {max1:.6g} [mm]') # correct
    print(f'second peak = {max2:.6g} [mm]') # correct
    print(f'time between peaks = {Td:.3g} [s]') # correct
    print(f'damping ratio = {damping_ratio:.3g}') # correct

    # 1c) Determine the mass of the piston.
    m = k / omega_n ** 2
    print(f'mass = {m:.3g} [kg]') # correct

if E3:
    ###############
    ## Problem 3 ##
    ###############
    print('\n', 15 * '=', '\n== Problem 3 == \n', 15 * '=', sep='')
    # Experimental conditions
    Re_x = 1e6
    Pr = 0.7
    x = 1.2 # [m]
    k = 0.0235 # [W/mK]
    nu = 11.42e-6 # [m^2/s]

    # Data
    measured_h = np.genfromtxt('data/heatsignal_2024.dat', skip_header=1)
    time = np.linspace(0.001, 1, 1000)
    ave_sig = np.average(measured_h)

    # Expressions 1 & 2
    Nu_x_1 = 0.029 * Re_x**0.8 * Pr**0.43
    C_f = 0.0592 * Re_x ** -0.2
    Nu_x_2 = (C_f/2 * Re_x * Pr) / (1 + 12.7 * (C_f/2)**0.5 * (Pr**(2/3) - 1))

    h_1 = Nu_x_1 * k / x
    h_2 = Nu_x_2 * k / x

    # determine independent samples
    i_arr, autocorfunc = get_Cxx(10, measured_h)

    plt.plot(i_arr, autocorfunc)
    plt.grid()
    plt.title('this')
    plt.show()
    # ==> samples are already independent --> NO, delta i = 10

    Sx_E3 = Sx(measured_h)
    Sx_bar_E3 = Sx_E3 / np.sqrt(len(measured_h)/10)

    delta_1 = abs(h_1 - ave_sig)
    delta_2 = abs(h_2 - ave_sig)

    test1 = test2 = 'Accepted'
    if delta_1 > 3*Sx_bar_E3:
        test1 = 'Rejected'

    if delta_2 > 3*Sx_bar_E3:
        test2 = 'Rejected'

    print(f'{test1} expression 1: h1 = {h_1:.3g} while average measured h = {ave_sig:.3g}')
    print(f'{test2} expression 2: h2 = {h_2:.3g} while average measured h = {ave_sig:.3g}')

if E4:
    ###############
    ## Problem 4 ##
    ###############
    print('\n', 15 * '=', '\n== Problem 4 ==\n', 15 * '=', sep='')

    # 4a) Determine the total uncertainty on x r at 95% confidence level
    # 95% confidence --> 2Sx
    u_resolution = 100
    u_offset = 50
    u_total = np.sqrt(u_resolution**2 + u_offset**2)
    print(f'total uncertainty = {u_total:.3g}')
    t_ref = -30

    # 4b) Determine the total uncertainty on x p at 95% confidence level for t = 10 min
    # and t = 60 min.

    # partial derivatives:
    ddxr_0 = lambda t: 1 - t/t_ref
    ddxr_t_ref = lambda t: t/t_ref

    u_1_10 = ddxr_0(10) * u_resolution
    u_2_10 = ddxr_t_ref(10) * u_resolution
    u_total_10 = np.sqrt(u_1_10 ** 2 + u_2_10 ** 2) # missing u_offset

    u_1_60 = ddxr_0(60) * u_resolution
    u_2_60 = ddxr_t_ref(60) * u_resolution
    u_total_60 = np.sqrt(u_1_60 ** 2 + u_2_60 ** 2) # missing u_offset

    print(f'At t=10 min --> u_xp = {u_total_10:.3g} [m]')
    print(f'At t=60 min --> u_xp = {u_total_60:.3g} [m]')

if E5:
    ###############
    ## Problem 5 ##
    ###############
    print('\n', 15 * '=', '\n== Problem 5 ==\n', 15 * '=', sep='')
    signal = np.genfromtxt('data/signal_x_2024.dat', skip_header=1)
    i_arr, autocorfunc = get_Cxx(15, signal)
    plt.plot(i_arr, autocorfunc)
    plt.grid()
    plt.show()

    # 5a) the variance of the noise
    average_sig = np.average(signal)
    Ssig = Sx(signal)
    i_independent = np.where(np.array(autocorfunc) < 0.05)[0][0]
    print(f'timespteps for independence = {i_independent}') # correct
    print(f'variance of the noise = Ssig = {Ssig:.3g}')

    # 5b) the actual variance of the physical variable x, | extrapolate the autocorrelation to zero to remove noise
    print(f'variance of the physical variable = Ssig / sqrt(N) = {Ssig/np.sqrt(len(signal)):.3g}')

    # 5c) the number of independent samples.
    print(f'number of independent signal = {math.floor(len(signal) / i_independent)}') # correct
    # 5d) Generate a random signal of the same length as x i and plot the correlation
    # function for this random signal.

    random_signal = np.random.rand(len(signal))
    corfunc = []

    for i in range(15):
        arr1 = signal[:len(signal)-i]
        arr2 = random_signal[i:]
        corfunc.append(cor(arr1, arr2))
    shift = [i for i in range(15)]
    plt.plot(shift, corfunc)
    plt.grid()
    plt.show()
