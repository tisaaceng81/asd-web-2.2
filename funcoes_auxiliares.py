import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from control import tf, step_response, feedback, bode, nyquist
import os

# ===========================
# Funções para EDO -> Função de Transferência
# ===========================

def parse_edo(edo_str, entrada, saida):
    """
    Recebe uma EDO em string e retorna:
    - Ls_expr: a função de Laplace transformada (função de transferência simbólica)
    - FT: função de transferência (objeto control.TransferFunction)
    - has_symbolic_coeffs: se há coeficientes simbólicos
    """
    edo_str = edo_str.replace("^", "**")  # caso o usuário use ^
    
    if '=' not in edo_str:
        raise ValueError("A EDO precisa ter um '=' separando LHS e RHS.")

    lhs_str, rhs_str = edo_str.split('=')
    
    try:
        t = sp.symbols('t')
        u = sp.Function(entrada)(t)
        y = sp.Function(saida)(t)
        
        lhs = sp.parse_expr(lhs_str, local_dict={entrada: u, saida: y, 'diff': sp.Derivative})
        rhs = sp.parse_expr(rhs_str, local_dict={entrada: u, saida: y, 'diff': sp.Derivative})
        
        edo_expr = lhs - rhs
        
        s = sp.symbols('s')
        Y = sp.Function('Y')(s)
        U = sp.Function('U')(s)

        subs_dict = {}
        for n in range(1, 10):  # suporta derivadas de qualquer ordem
            subs_dict[sp.Derivative(y, (t, n))] = s**n * Y
            subs_dict[sp.Derivative(u, (t, n))] = s**n * U
        subs_dict[y] = Y
        subs_dict[u] = U

        Ls_expr = edo_expr.subs(subs_dict)
        
        # Verifica se há coeficientes simbólicos além de 's'
        has_symbolic_coeffs = any(isinstance(c, sp.Symbol) for c in Ls_expr.atoms(sp.Symbol) if str(c) not in ['s'])
        
        # Placeholder para função de transferência
        FT = sp.simplify(Y / U)
        
        return Ls_expr, FT, has_symbolic_coeffs

    except Exception as e:
        raise ValueError(f"Erro ao processar a EDO: {e}")

def ft_to_latex(ft_expr):
    return sp.latex(ft_expr, mul_symbol='dot')

# ===========================
# Resposta ao Degrau
# ===========================

def resposta_degrau(FT):
    t = np.linspace(0, 10, 1000)
    T = tf([1], [1, 0])
    yout, t = step_response(T)
    return t, yout

def salvar_grafico_resposta(t, y, nome_arquivo, deslocamento=0.0):
    plt.figure(figsize=(6,4))
    plt.plot(t, y + deslocamento, 'b', lw=2)
    plt.xlabel('Tempo [s]')
    plt.ylabel('Saída')
    plt.title('Resposta ao Degrau')
    plt.grid(True)
    path = f'static/{nome_arquivo}.png'
    plt.savefig(path)
    plt.close()
    return path

# ===========================
# Estima L e T
# ===========================

def estima_LT(t, y):
    # Método simplificado para estimar L e T
    L = 0.1
    T = 1.0
    return L, T

# ===========================
# Sintonia PID
# ===========================

def sintonia_ziegler_nichols(L, T):
    Kp = 1.2*T/L
    Ki = 2*L
    Kd = 0.5*L
    return Kp, Ki, Kd

def sintonia_oscilacao_forcada(Kc, Tc):
    Kp = 0.6*Kc
    Ki = 1.2*Kc/Tc
    Kd = 0.075*Kc*Tc
    return Kp, Ki, Kd

def cria_pid_tf(Kp, Ki, Kd):
    return tf([Kd, Kp, Ki], [1, 0])

def malha_fechada_tf(plant_tf, pid_tf):
    loop = pid_tf * plant_tf
    closed = feedback(loop, 1)
    return closed

# ===========================
# Polos e Zeros
# ===========================

def plot_polos_zeros(FT):
    plt.figure(figsize=(6,6))
    plt.title('Polos e Zeros')
    plt.grid(True)
    # Dummy plot
    plt.scatter([-1], [-1], color='red')
    path = 'static/polos_zeros.png'
    plt.savefig(path)
    plt.close()
    return path

# ===========================
# Routh-Hurwitz
# ===========================

def flatten_and_convert(den):
    return [float(c) for c in den]

def tabela_routh(coefs):
    n = len(coefs)
    table = np.zeros((n, n//2 + n%2))
    table[0,:len(coefs[::2])] = coefs[::2]
    table[1,:len(coefs[1::2])] = coefs[1::2]
    for i in range(2, n):
        for j in range(table.shape[1]-1):
            a = table[i-2,0]
            b = table[i-2,j+1] if j+1 < table.shape[1] else 0
            c = table[i-1,0]
            d = table[i-1,j+1] if j+1 < table.shape[1] else 0
            if c != 0:
                table[i,j] = (c*b - a*d)/c
            else:
                table[i,j] = 0
    return table
