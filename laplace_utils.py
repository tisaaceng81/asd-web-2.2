import sympy as sp
import matplotlib.pyplot as plt
import control
import numpy as np
from control.matlab import step

# --- Função para parsear EDO simbólica e gerar FT malha aberta ---
def parse_edo(edo_str, entrada_str, saida_str):
    try:
        t = sp.symbols('t', real=True)
        s = sp.symbols('s')
        x = sp.Function(saida_str)(t)
        F = sp.Function(entrada_str)(t)

        # Substitui diff() por sp.Derivative()
        eq_str = edo_str.replace('diff', 'sp.Derivative')
        if '=' in eq_str:
            lhs, rhs = eq_str.split('=')
            eq_str = f"({lhs.strip()}) - ({rhs.strip()})"

        local_dict = {
            'sp': sp,
            't': t,
            entrada_str: F,
            saida_str: x,
            'u': F,
            'y': x,
            str(F): F,
            str(x): x
        }

        eq = sp.sympify(eq_str, locals=local_dict)

        Xs, Fs = sp.symbols('Xs Fs')
        expr_laplace = eq

        # Substituir derivadas por s^n * Xs ou Fs
        for d in expr_laplace.atoms(sp.Derivative):
            ordem = d.derivative_count
            func = d.expr
            if func == x:
                expr_laplace = expr_laplace.subs(d, s**ordem * Xs)
            elif func == F:
                expr_laplace = expr_laplace.subs(d, s**ordem * Fs)

        # Substituir as funções por variáveis de Laplace
        expr_laplace = expr_laplace.subs({
            x: Xs,
            F: Fs,
            sp.Function(entrada_str)(t): Fs,
            sp.Function(saida_str)(t): Xs
        })

        lhs = expr_laplace
        coef_Xs = lhs.coeff(Xs)
        resto = lhs - coef_Xs * Xs
        Ls_expr = -resto / coef_Xs
        Ls_expr = sp.simplify(Ls_expr)

        Ls_expr = Ls_expr.subs(Fs, 1)  # ✅ Substitui Fs por 1 para obter FT numérica

        num, den = sp.fraction(Ls_expr)
        num_poly = sp.Poly(num, s)
        den_poly = sp.Poly(den, s)

        # ✅ Evita erro de símbolo simbólico na conversão
        def safe_float_list(coeffs):
            result = []
            for c in coeffs:
                c_eval = c.evalf()
                if not c_eval.free_symbols:
                    result.append(float(c_eval))
                else:
                    raise Exception(f"Coeficiente não numérico encontrado: {c}")
            return result

        num_coeffs = safe_float_list(num_poly.all_coeffs())
        den_coeffs = safe_float_list(den_poly.all_coeffs())

        FT = control.TransferFunction(num_coeffs, den_coeffs)
        return Ls_expr, FT

    except Exception as e:
        raise Exception(f"Erro ao interpretar a EDO: {str(e)}")


def ft_to_latex(expr):
    return sp.latex(expr, mul_symbol='dot')


def resposta_degrau(FT, tempo=None):
    if tempo is None:
        tempo = np.linspace(0, 10, 1000)
    t, y = step(FT, T=tempo)
    return t, y


def estima_LT(t, y):
    y_final = y[-1]
    indice_inicio = next(i for i, v in enumerate(y) if v > 0.01 * y_final)
    L = t[indice_inicio]
    y_63 = 0.63 * y_final
    indice_63 = next(i for i, v in enumerate(y) if v >= y_63)
    T = t[indice_63] - L
    if T < 0:
        T = 0.01
    return L, T


def sintonia_ziegler_nichols(L, T):
    Kp = 1.2 * T / L
    Ti = 2 * L
    Td = 0.5 * L
    Ki = Kp / Ti
    Kd = Kp * Td
    return Kp, Ki, Kd


def cria_pid_tf(Kp, Ki, Kd):
    s = control.TransferFunction.s
    return Kp + Ki / s + Kd * s


def malha_fechada_tf(Gp, Gc):
    return control.feedback(Gp * Gc, 1)


def tabela_routh(coeficientes):
    # ✅ Correção para tratar listas aninhadas no denominador
    coeficientes = [float(c[0]) if isinstance(c, list) else float(c) for c in coeficientes]

    n = len(coeficientes)
    m = (n + 1) // 2
    routh = np.zeros((n, m))

    routh[0, :len(coeficientes[::2])] = coeficientes[::2]
    if n > 1:
        routh[1, :len(coeficientes[1::2])] = coeficientes[1::2]

    for i in range(2, n):
        for j in range(m - 1):
            a = routh[i - 2, 0]
            b = routh[i - 1, 0]
            c = routh[i - 2, j + 1]
            d = routh[i - 1, j + 1]
            if b == 0:
                b = 1e-6
            routh[i, j] = (b * c - a * d) / b
    return routh


def plot_polos_zeros(FT):
    plt.figure()
    poles = FT.poles()
    zeros = FT.zeros()
    plt.scatter(np.real(poles), np.imag(poles), marker='x', color='red', label='Polos')
    plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='blue', label='Zeros')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title('Polos e Zeros')
    plt.legend()
    plt.grid(True)
    plt.show()