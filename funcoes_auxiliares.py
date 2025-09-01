import sympy as sp
import control
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from sympy.abc import s, K
from sympy import *

def flatten_and_convert(lst):
    result = []
    for c in lst:
        if hasattr(c, '__iter__') and not isinstance(c, (str, bytes)):
            result.extend(flatten_and_convert(c))
        else:
            try:
                result.append(float(c))
            except Exception as e:
                raise Exception(f"Erro convertendo coeficiente para número: {c} ({e}).")
    return result

def pad_coeffs(num_coeffs, den_coeffs):
    len_num = len(num_coeffs)
    len_den = len(den_coeffs)
    if len_num < len_den:
        num_coeffs = [0.0] * (len_den - len_num) + num_coeffs
    elif len_den < len_num:
        den_coeffs = [0.0] * (len_num - len_den) + den_coeffs
    return num_coeffs, den_coeffs

def parse_edo(edo_str, entrada_str, saida_str):
    t = sp.symbols('t', real=True)
    X_func = sp.Function(saida_str)
    F_func = sp.Function(entrada_str)

    # Verifica se a sintaxe é 'diff' e substitui por 'sp.Derivative'
    eq_str = edo_str.replace('diff', 'sp.Derivative')
    
    if '=' not in eq_str:
        raise ValueError("A EDO deve conter '=' para separar LHS e RHS.")
    
    lhs, rhs = eq_str.split('=')
    eq_str = f"({lhs.strip()}) - ({rhs.strip()})"

    potential_symbols = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', eq_str))
    local_dict = {'sp': sp, 't': t, saida_str: X_func, entrada_str: F_func, 'Derivative': sp.Derivative}
    excluded = {'t', 'diff', 'sp', 'Derivative', entrada_str, saida_str}
    
    for sym in potential_symbols:
        if sym not in excluded and sym not in local_dict:
            local_dict[sym] = sp.symbols(sym)

    eq = sp.sympify(eq_str, locals=local_dict)
    
    # Verifica se a EDO contém a variável de saída
    if not any(f.func == X_func for f in eq.atoms(sp.Function)):
        raise ValueError(f"Lado esquerdo da EDO deve conter a variável de saída '{saida_str}(t)'.")

    Xs = sp.Symbol(f'{saida_str}s')
    Fs = sp.Symbol(f'{entrada_str}s')
    
    expr_laplace = eq
    
    # Substitui as derivadas por seus equivalentes em Laplace.
    derivatives = list(expr_laplace.atoms(sp.Derivative))
    derivatives.sort(key=lambda d: d.derivative_count, reverse=True)
    
    for d in derivatives:
        order = d.derivative_count
        func_expr = d.args[0]
        if func_expr.func == X_func:
            expr_laplace = expr_laplace.subs(d, sp.Symbol('s')**order * Xs)
        elif func_expr.func == F_func:
            expr_laplace = expr_laplace.subs(d, sp.Symbol('s')**order * Fs)

    # Substitui as funções por suas variáveis de Laplace
    expr_laplace = expr_laplace.subs({X_func(t): Xs, F_func(t): Fs})

    collected_expr = sp.collect(expr_laplace, [Xs, Fs])
    coef_Xs = collected_expr.coeff(Xs)
    coef_Fs = collected_expr.coeff(Fs)

    if coef_Xs == 0:
        raise ValueError(f"Coeficiente da variável de saída no domínio de Laplace é zero.")

    Ls_expr = -coef_Fs / coef_Xs
    Ls_expr = sp.simplify(Ls_expr)

    num, den = sp.fraction(Ls_expr)
    num_poly = sp.Poly(num, sp.Symbol('s'))
    den_poly = sp.Poly(den, sp.Symbol('s'))

    has_symbolic = False
    try:
        num_coeffs = [float(c) for c in num_poly.all_coeffs()]
        den_coeffs = [float(c) for c in den_poly.all_coeffs()]
    except Exception:
        has_symbolic = True
        return Ls_expr, None, True

    num_coeffs, den_coeffs = pad_coeffs(num_coeffs, den_coeffs)

    FT = control.TransferFunction(num_coeffs, den_coeffs)
    return Ls_expr, FT, False

def calcula_kc_tc_automaticamente(den_coeffs):
    """
    Calcula Kc e Tc usando o critério de Routh-Hurwitz para uma FT de 3ª ordem.
    (A função pode ser estendida para outras ordens)
    """
    # Exemplo para FT de 3ª ordem. Adapte a ordem para o seu uso.
    a3, a2, a1, a0 = den_coeffs
    
    kc = (a2 * a1 - a0) / 1.0
    tc = 2*pi / (a1 / 1.0)**(1/2) # Adapte a formula do TC para a sua FT
    
    return kc, tc

def ft_to_latex(expr):
    return sp.latex(expr, mul_symbol='dot')

def resposta_degrau(FT, tempo=None):
    if tempo is None:
        tempo = np.linspace(0, 10, 1000)
    t, y = control.step_response(FT, T=tempo)
    return t, y

def estima_LT(t, y):
    if len(y) == 0 or np.isclose(y[-1], 0):
        return 0.01, 0.01
    y_final = y[-1]
    if y_final == 0:
        return 0.01, 0.01
    try:
        indice_inicio = next(i for i, v in enumerate(y) if v > 0.01 * y_final)
    except StopIteration:
        indice_inicio = 0
    L = t[indice_inicio]
    y_63 = 0.63 * y_final
    try:
        indice_63 = next(i for i, v in enumerate(y) if v >= y_63)
    except StopIteration:
        indice_63 = len(y) - 1
    T = t[indice_63] - L
    return max(L, 0.01), max(T, 0.01)

def sintonia_ziegler_nichols(L, T):
    if L == 0:
        L = 0.01
    Kp = 1.2 * T / L
    Ti = 2 * L
    Td = 0.5 * L
    Ki = Kp / Ti if Ti != 0 else 0
    Kd = Kp * Td
    return Kp, Ki, Kd

def sintonia_oscilacao_forcada(Kc, Tc):
    # PID Clássico
    Kp = 0.6 * Kc
    Ti = Tc / 2.0
    Td = Tc / 8.0
    
    Ki = Kp / Ti if Ti != 0 else 0
    Kd = Kp * Td

    return Kp, Ki, Kd

def cria_pid_tf(Kp, Ki, Kd):
    s = control.TransferFunction.s
    return Kp + Ki / s + Kd * s

def malha_fechada_tf(Gp, Gc):
    return control.feedback(Gp * Gc, 1)

def salvar_grafico_resposta(t, y, nome, rotacao=0, deslocamento=0.0):
    y = np.array(y) + deslocamento
    if rotacao == 180:
        t = -t[::-1]
        y = -y[::-1]
    elif rotacao == 90:
        t, y = -y, t
    if min(t) < 0:
        t = t - min(t)
    if min(y) < 0:
        y = y - min(y)
    plt.figure(figsize=(8, 4))
    plt.plot(t, y, label='Resposta ao Degrau')
    plt.xlabel('Tempo (s)' if rotacao != 90 else 'Saída')
    plt.ylabel('Saída' if rotacao != 90 else 'Tempo (s)')
    plt.title(nome)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    caminho = os.path.join('static', f'{nome}.png')
    plt.savefig(caminho)
    plt.close()
    return caminho

def plot_polos_zeros(FT):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(np.real(FT.poles()), np.imag(FT.poles()), marker='x', color='red', s=100, label='Polos')
    ax.scatter(np.real(FT.zeros()), np.imag(FT.zeros()), marker='o', color='blue', s=100, facecolors='none', edgecolors='blue', label='Zeros')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_title('Diagrama de Polos e Zeros')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')
    caminho = os.path.join('static', 'polos_zeros.png')
    plt.savefig(caminho)
    plt.close()
    return caminho

def tf_to_sympy_tf(tf_obj):
    import sympy as sp
    num = tf_obj.num[0][0]
    den = tf_obj.den[0][0]
    s_sym = sp.symbols('s')
    num_poly = sum(coef * s_sym**(len(num)-i-1) for i, coef in enumerate(num))
    den_poly = sum(coef * s_sym**(len(den)-i-1) for i, coef in enumerate(den))
    return num_poly / den_poly

def tabela_routh(coeficientes):
    coeficientes = [float(c[0]) if isinstance(c, list) else float(c) for c in coeficientes]
    n = len(coeficientes)
    if n == 0:
        return np.array([])
    m = (n + 1) // 2
    routh = np.zeros((n, m))
    routh[0, :len(coeficientes[::2])] = coeficientes[::2]
    if n > 1:
        routh[1, :len(coeficientes[1::2])] = coeficientes[1::2]
    for i in range(2, n):
        if routh[i - 1, 0] == 0:
            routh[i - 1, 0] = 1e-6
        for j in range(m - 1):
            a = routh[i - 2, 0]
            b = routh[i - 1, 0]
            c = routh[i - 2, j + 1]
            d = routh[i - 1, j + 1]
            routh[i, j] = (b * c - a * d) / b
    return routh
