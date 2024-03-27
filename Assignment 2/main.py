from tabulate import tabulate, SEPARATING_LINE
import numpy as np
import math
pt = [['Problem', 'Parameter', 'Value', 'Unit']]

execute_problem1 = True
execute_problem2 = True
execute_problem3 = True

# ======================================================================
# =                           Problem 1                                =
# ======================================================================
if execute_problem1:

    # ================= 1 (a) ======================
    R0_p1 = 10                 # [Ohm]
    T0_p1 = 20                 # [C]
    alpha = 0.00847         # [1/K]
    Twire = 150             # [K]
    R4_p1 = 70                 # [Ohm]
    R3_p1 = 40                 # [Ohm]

    Rs = R0_p1 * (1 + alpha * (Twire - T0_p1))    # [Ohm]
    R_D = R4_p1 * (Rs / R3_p1)                    # [Ohm]
    pt.append(['1a', 'Rd', R_D, 'Ohm'])

    # ================= 1 (b) ======================
    heat_loss = 0.0113                  # [W]
    I_s3 = math.sqrt(heat_loss / Rs)    # [A]
    Em_p1 = I_s3 * (Rs + R3_p1)               # [V]
    pt.append(['1b', 'Em', Em_p1, 'V'])

    # ================= 1 (c) ======================
    # Equal to Em

    # ================= 1 (d) ======================
    # Choose filter A: 4kHz and k=1 since Nyquist frequency is 5 kHz
    # Should not capture any greater frequencies than Nyquist frequency

    # ================= 1 (e) ======================
    Asig = 0.29                             # [m/s]
    zeta = 0.4                              # [-]
    omega_n = 19 * 2 * math.pi              # [rad/s]
    omega_sig = 12 * 2 * math.pi            # [rad/s]
    omega_ratio = omega_sig / omega_n
    sys_gain = 1 / math.sqrt((1 - omega_ratio ** 2) ** 2 + (2 * zeta * omega_ratio) ** 2)
    filter_gain = 1 / math.sqrt(1 + (omega_sig * 3.97 * 1e-5) ** 2)
    A_out = Asig * sys_gain * filter_gain   # [m/s]
    pt.append(['1e', 'A out', A_out, 'm/s'])

    # ================= 1 (f) ======================
    # bell curve with mean=5, sigma = 3.5

# ======================================================================
# =                           Problem 2                                =
# ======================================================================
if execute_problem2:

    # ================= 2 (a) ======================
    R1_p2 = 4000                   # [Ohm]
    R2_p2 = 7000                   # [Ohm]
    R3_p2 = 5000                   # [Ohm]
    E_i = 15                     # [V]

    R4_p2 = R3_p2 * (R2_p2 / R1_p2)         # [Ohm]
    pt.append(['2a', 'R4', R4_p2, 'Ohm'])

    # ================= 2 (b) ======================
    Em_p2 = -0.0037                  # [V]
    R_m = 80000
    I_m = Em_p2 / R_m
    GF = 2                           # [?]
    # dR  = R * (dL/L) * GF

    C0 = -1 * Em_p2 + R3_p2 * E_i / (R3_p2 + R4_p2)
    R1_msm_2b = R2_p2 * C0 / (E_i - C0)
    epsilon_2b = (R1_msm_2b - R1_p2) / (R1_p2 * GF)
    pt.append(['2b', 'R1', R1_msm_2b, 'Ohm'])
    pt.append(['2b', 'epsilon', epsilon_2b, '-'])

    # ================= 2 (c) ======================
    # eq1: I2 = I1 - Im
    # eq2: I4 = I3 + Im
    # eq3: EB = Ei - I1 * R1
    # eq4: EC = Ei - I3 * R3
    # eq5: EB - EC = Em
    # eq6: EB = I2 * R2
    # eq7: EC = I4 * R4

    C1 = E_i + R2_p2 * I_m
    C2 = Em_p2 - (E_i - R4_p2 * I_m) * R3_p2 / (R3_p2 + R4_p2)
    R1_msm_2c = -1 * R2_p2 * C2 / (C1 + C2)

    pt.append(['2c', 'R1', R1_msm_2c, 'Ohm'])
    epsilon_2c = (R1_msm_2c - R1_p2) / (R1_p2 * GF)
    pt.append(['2c', 'epsilon', epsilon_2c, '-'])

    # ================= 2 (d) ======================
    loading_error = (epsilon_2c - epsilon_2b)/epsilon_2b * 100
    pt.append(['2d', 'loading error', loading_error, '%'])

    # ================= 2 (e) ======================
    # Calibrate

# ======================================================================
# =                           Problem 3                                =
# ======================================================================
if execute_problem3:

    # ================= 3 (a) ======================
    pass




print(tabulate(pt))
