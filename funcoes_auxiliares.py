import sympy as sp
import control
import numpy as np
import matplotlib.pyplot as plt
import os

# Cria diretório estático se não existir
STATIC_DIR = os.path.join(os.getcwd(), 'static')
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

def parse_edo(edo_str, entrada, saida):
    """
    Converte uma string de EDO em função de t para uma função de transferência.
    Suporta derivadas de qualquer ordem.
    """
    t = sp.symbols('t')
    s = sp.symbols('s')
    y = sp.Function(saida)(t)
    u = sp.Function(entrada)(t)

    # Converte a string em expressão simbólica
    expr = sp.sympify(edo_str, locals={saida: y, entrada: u, 'diff': sp.diff})

    # Extrai coeficientes numéricos ou simbólicos
    eq = sp.expand(expr)
    coeffs_y = []
    max_order = 0

    # Determina a maior ordem de derivada
    for arg in eq.args if hasattr(eq, 'args') else [eq]:
        if arg.has(sp.Derivative(y, t)):
            for deriv in arg.atoms(sp.Derivative):
                if deriv.args[0] == y:
                    max_order = max(max_order, deriv.derivative_count if hasattr(deriv, 'derivative_count') else deriv.args[1])

    # Cria vetor de coeficientes da EDO (numeradores e denominadores)
    coefs_num = []
    coefs_den = []

    # Para cada ordem, extrai coeficiente de y^(n) e u^(n)
    for n in range(max_order, -1, -1):
        deriv_y = sp.Derivative(y, (t, n))
        coef_y = eq.coeff(deriv_y)
        coeffs_y.append(coef_y)

    deriv_u = sp.Derivative(u, t, 0)
    num_expr = eq.subs({sp.Derivative(y, (t, n)): 0 for n in range(max_order+1)})
    num_expr = num_expr.coeff(u)
    
    # Gera função de transferência simbólica
    s = sp.symbols('s')
    num_poly = sum(coef * s**i for i, coef in enumerate(reversed([num_expr])))
    den_poly = sum(coef * s**i for i, coef in enumerate(reversed(coeffs_y)))
    FT = control.TransferFunction([float(c.evalf()) if c.is_number else c for c in den_poly.as_coefficients_dict().values()],
                                  [float(c.evalf()) if c.is_number else c for c in den_poly.as_coefficients_dict().values()])

    has_symbolic_coeffs = any([not c.is_number for c in coeffs_y])
    return FT, FT, has_symbolic_coeffs

def ft_to_latex(ft):
    """Converte uma função de transferência para LaTeX"""
    num = ft.num[0][0]
    den = ft.den[0][0]
    s = sp.symbols('s')
    num_poly = sum(coef * s**(len(num)-i-1) for i, coef in enumerate(num))
    den_poly = sum(coef * s**(len(den)-i-1) for i, coef in enumerate(den))
    return sp.latex(num_poly/den_poly, mul_symbol='dot')

def resposta_degrau(FT):
    """Gera resposta ao degrau"""
    t, y = control.step_response(FT)
    return t, y

def estima_LT(t, y):
    """Estima L e T a partir da resposta ao degrau"""
    y_final = y[-1]
    y_63 = 0.632 * y_final
    idx_63 = np.argmax(y >= y_63)
    T = t[idx_63]
    L = t[0]  # Aproximação inicial
    return L, T

def sintonia_ziegler_nichols(L, T):
    """Calcula parâmetros PID via Ziegler-Nichols (resposta ao degrau)"""
    Kp = 1.2 * T / L
    Ti = 2 * L
    Td = 0.5 * L
    Ki = Kp / Ti
    Kd = Kp * Td
    return Kp, Ki, Kd

def sintonia_oscilacao_forcada(Kc, Tc):
    """Calcula parâmetros PID via oscilação forçada"""
    Kp = 0.6 * Kc
    Ki = 1.2 * Kp / Tc
    Kd = 0.075 * Kp * Tc
    return Kp, Ki, Kd

def cria_pid_tf(Kp, Ki, Kd):
    """Gera função de transferência do PID"""
    s = control.TransferFunction.s
    return Kp + Ki/s + Kd*s

def malha_fechada_tf(FT, PID):
    """Gera a malha fechada com PID"""
    return control.feedback(PID * FT, 1)

def salvar_grafico_resposta(t, y, nome, deslocamento=0.0):
    """Salva gráfico de resposta ao degrau"""
    plt.figure()
    plt.plot(t, y + deslocamento)
    plt.title('Resposta ao Degrau')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Saída')
    caminho = os.path.join(STATIC_DIR, f'{nome}.png')
    plt.savefig(caminho)
    plt.close()
    return caminho

def plot_polos_zeros(FT):
    """Gera gráfico de polos e zeros"""
    plt.figure()
    control.pzmap(FT, Plot=True, grid=True)
    caminho = os.path.join(STATIC_DIR, 'pz_map.png')
    plt.savefig(caminho)
    plt.close()
    return caminho

def flatten_and_convert(lst):
    """Flatten lista e converte para float"""
    return [float(x) for sublist in lst for x in sublist]

def tabela_routh(coefs):
    """Gera tabela de Routh-Hurwitz"""
    n = len(coefs)
    routh = np.zeros((n, int(np.ceil(n/2))))
    routh[0, :len(coefs[::2])] = coefs[::2]
    routh[1, :len(coefs[1::2])] = coefs[1::2]

    for i in range(2, n):
        for j in range(0, routh.shape[1]-1):
            a = routh[i-2,0]
            b = routh[i-2,j+1]
            c = routh[i-1,0]
            d = routh[i-1,j+1]
            routh[i,j] = (c*b - a*d)/c if c != 0 else 0
    return routh
