import numpy as np
import matplotlib.pyplot as plt


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
plt.legend()
plt.show()

rho_ideal_gas_T_ave = p_arr / (R * T_ave)
true_A = 1 / (R*T_ave)

# 1k) error A and Systematic error B
print(f'\n[N-SHUFFLED] error A = {p_vs_rho_linfit[0] - true_A:.3g} error B = {p_vs_rho_linfit[1]:.3g}')
print(f'[SHUFFLED] error A = {p_vs_rho_linfit_sh[0] - true_A:.3g} error B = {p_vs_rho_linfit_sh[1]:.3g}')

# 1l) because the random spread is larger when shuffling
