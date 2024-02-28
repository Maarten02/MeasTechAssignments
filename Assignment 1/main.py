import numpy as np
import matplotlib.pyplot as plt


E1 = False
E2 = True

if E1:
    ###############
    ## Problem 1 ##
    ###############
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
    standard_error = np.sum((rho_arr - rho_fit) ** 2) / (401 - (1 + 1))
    print(f'Standard error = {standard_error:.3g}')

    # 1f)
    confidence_uppend = 2 * standard_error * np.sqrt(1/401 + (150000 - 100000) / np.sum((p_arr - 100000)**2))
    print(f'95% confidence on upper end of interval  = (+/-) {confidence_uppend:.3g}')

    # 1g)
    shuffled_indices = np.random.permutation(len(T_arr))
    shuffled_T = T_arr[shuffled_indices]
    rho_arr_sh = p_arr / (R * shuffled_T)

    # 1h)
    p_vs_rho_linfit_sh = np.polyfit(p_arr, rho_arr_sh, 1)
    print(f'\n[SHUFFLED] Linear fit rho = {p_vs_rho_linfit_sh[0]:.6g}p + {p_vs_rho_linfit_sh[1]:.4g}')
    rho_fit_sh = p_arr * p_vs_rho_linfit_sh[0] + p_vs_rho_linfit_sh[1]
    ax.plot(p_arr, rho_fit_sh, label='Linear fit [SHUF]')

    # 1i)
    standard_error_sh = np.sum((rho_arr_sh - rho_fit_sh) ** 2) / (401 - (1 + 1))
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
    m = k / omega_n ** 2

    print(f'natural frequency = {1/T_n:.3g} [Hz]')
    print(f'first peak = {max1:.6g} [mm]')
    print(f'second peak = {max2:.6g} [mm]')
    print(f'time between peaks = {Td:.3g} [s]')
    print(f'damping ratio = {damping_ratio:.3g}')
    print(f'mass = {m:.3g} [kg]')

    # 1c) Determine the mass of the piston.