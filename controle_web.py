
     from flask import Flask, render_template, request, redirect, url_for, flash, session
import json
import os
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import control
from control.matlab import step
from sympy.abc import s

app = Flask(__name__)
app.secret_key = 'sua_chave_secreta'

# === FUNÇÕES AUXILIARES ===

def flatten_and_convert(lst):
    result = []
    for c in lst:
        if hasattr(c, '__iter__') and not isinstance(c, (str, bytes)):
            result.extend(flatten_and_convert(c))
        else:
            result.append(sp.sympify(c))
    return result

def pad_coeffs(num_coeffs, den_coeffs):
    len_num = len(num_coeffs)
    len_den = len(den_coeffs)
    if len_num < len_den:
        num_coeffs = [0] * (len_den - len_num) + num_coeffs
    elif len_den < len_num:
        den_coeffs = [0] * (len_num - len_den) + den_coeffs
    return num_coeffs, den_coeffs

def parse_edo(edo_str, entrada_str, saida_str):
    t = sp.symbols('t', real=True)
    y = sp.Function(saida_str)(t)
    u = sp.Function(entrada_str)(t)

    eq_str = edo_str.replace('diff', 'sp.Derivative')
    if '=' in eq_str:
        lhs, rhs = eq_str.split('=')
        eq_str = f"({lhs.strip()}) - ({rhs.strip()})"

    local_dict = {"sp": sp, "t": t, entrada_str: u, saida_str: y, str(u): u, str(y): y}

    eq = sp.sympify(eq_str, locals=local_dict)

    Ys, Us = sp.symbols('Ys Us')
    expr_laplace = eq
    for d in eq.atoms(sp.Derivative):
        ordem = d.derivative_count
        func = d.expr
        if func == y:
            expr_laplace = expr_laplace.subs(d, s**ordem * Ys)
        elif func == u:
            expr_laplace = expr_laplace.subs(d, s**ordem * Us)

    expr_laplace = expr_laplace.subs({y: Ys, u: Us})
    lhs = expr_laplace
    coef_Ys = lhs.coeff(Ys)
    resto = lhs - coef_Ys * Ys
    Ls_expr = -resto / coef_Ys
    Ls_expr = sp.simplify(Ls_expr.subs(Us, 1))

    num, den = sp.fraction(Ls_expr)
    num_coeffs = sp.Poly(num, s).all_coeffs()
    den_coeffs = sp.Poly(den, s).all_coeffs()
    num_coeffs, den_coeffs = pad_coeffs(num_coeffs, den_coeffs)

    num_coeffs_eval = [float(c.evalf()) for c in num_coeffs]
    den_coeffs_eval = [float(c.evalf()) for c in den_coeffs]

    FT = control.TransferFunction(num_coeffs_eval, den_coeffs_eval)
    return Ls_expr, FT

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
    return (L if L >= 0 else 0.01), T

def sintonia_ziegler_nichols(L, T):
    Kp = 1.2 * T / L if L != 0 else 1.0
    Ti = 2 * L
    Td = 0.5 * L
    Ki = Kp / Ti if Ti != 0 else 0.0
    Kd = Kp * Td
    return Kp, Ki, Kd

def cria_pid_tf(Kp, Ki, Kd):
    s = control.TransferFunction.s
    return Kp + Ki / s + Kd * s

def malha_fechada_tf(Gp, Gc):
    return control.feedback(Gp * Gc, 1)

def tabela_routh(coeficientes):
    coeficientes = [float(c) for c in coeficientes]
    n = len(coeficientes)
    m = (n + 1) // 2
    routh = np.zeros((n, m))
    routh[0, :len(coeficientes[::2])] = coeficientes[::2]
    if n > 1:
        routh[1, :len(coeficientes[1::2])] = coeficientes[1::2]
    for i in range(2, n):
        for j in range(m - 1):
            a = routh[i - 2, 0]
            b = routh[i - 1, 0] if routh[i - 1, 0] != 0 else 1e-6
            c = routh[i - 2, j + 1]
            d = routh[i - 1, j + 1]
            routh[i, j] = (b * c - a * d) / b
    return routh

def salvar_grafico_resposta(t, y, nome, rotacao=0, deslocamento=0.0):
    y = y + deslocamento

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
    fig, ax = plt.subplots()
    ax.scatter(np.real(FT.poles()), np.imag(FT.poles()), marker='x', color='red', label='Polos')
    ax.scatter(np.real(FT.zeros()), np.imag(FT.zeros()), marker='o', color='blue', label='Zeros')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_title('Diagrama de Polos e Zeros')
    ax.legend()
    ax.grid(True)
    caminho = os.path.join('static', 'polos_zeros.png')
    plt.savefig(caminho)
    plt.close()
    return caminho


# === ROTAS ===

@app.route('/')
def home():
    user_email = session.get('usuario_logado')
    return render_template('index.html', user_email=user_email)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        senha = request.form['senha']
        if os.path.exists('usuarios.json'):
            with open('usuarios.json', 'r') as f:
                usuarios = json.load(f)
        else:
            usuarios = {}

        # Admin fixo com email e senha definidos
        if email == 'tisaaceng@gmail.com' and senha == '4839AT81':
            session['usuario_logado'] = email
            session['is_admin'] = True
            return redirect(url_for('admin'))

        if email in usuarios and usuarios[email]['senha'] == senha:
            if usuarios[email].get('aprovado', False):
                session['usuario_logado'] = email
                session['is_admin'] = False
                return redirect(url_for('painel'))
            else:
                flash('Cadastro ainda não aprovado.')
                return redirect(url_for('login'))
        else:
            flash('Credenciais inválidas.')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/cadastro', methods=['GET', 'POST'])
def cadastro():
    if request.method == 'POST':
        nome = request.form['nome']
        email = request.form['email']
        senha = request.form['senha']
        if os.path.exists('usuarios.json'):
            with open('usuarios.json', 'r') as f:
                usuarios = json.load(f)
        else:
            usuarios = {}

        if email in usuarios or email == 'tisaaceng@gmail.com':
            flash('Email já cadastrado ou reservado.')
            return redirect(url_for('cadastro'))

        usuarios[email] = {'nome': nome, 'senha': senha, 'aprovado': False}
        with open('usuarios.json', 'w') as f:
            json.dump(usuarios, f, indent=4)

        flash('Cadastro enviado para aprovação.')
        return redirect(url_for('login'))
    return render_template('cadastro.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    # Permite acesso somente se estiver logado como admin (email fixo + flag)
    if 'usuario_logado' not in session or session.get('is_admin') != True:
        flash('Acesso negado. Apenas o administrador pode acessar.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        email = request.form.get('email')
        if os.path.exists('usuarios.json'):
            with open('usuarios.json', 'r') as f:
                usuarios = json.load(f)
            if email in usuarios:
                usuarios[email]['aprovado'] = True
                with open('usuarios.json', 'w') as f:
                    json.dump(usuarios, f, indent=4)
                flash(f'{email} aprovado com sucesso!')
        else:
            flash('Arquivo de usuários não encontrado.')

    # Mostrar apenas usuários não aprovados
    if os.path.exists('usuarios.json'):
        with open('usuarios.json', 'r') as f:
            usuarios = json.load(f)
        nao_aprovados = {k: v for k, v in usuarios.items() if not v.get('aprovado', False)}
    else:
        nao_aprovados = {}

    return render_template('admin.html', usuarios=nao_aprovados)

@app.route('/painel')
def painel():
    if 'usuario_logado' not in session:
        flash('Acesso negado.')
        return redirect(url_for('login'))
    return render_template('painel.html')

@app.route('/logout')
def logout():
    session.pop('usuario_logado', None)
    session.pop('is_admin', None)
    flash('Logout realizado.')
    return redirect(url_for('login'))

@app.route('/simulador', methods=['GET', 'POST'])
def simulador():
    if 'usuario_logado' not in session:
        flash('Faça login para acessar o simulador.')
        return redirect(url_for('login'))

    email = session['usuario_logado']

    # Verificar se usuário está aprovado
    if email != 'tisaaceng@gmail.com':  # Admin não precisa estar aprovado para acessar o simulador
        if os.path.exists('usuarios.json'):
            with open('usuarios.json', 'r') as f:
                usuarios = json.load(f)
            if not usuarios.get(email, {}).get('aprovado', False):
                flash('Seu cadastro ainda não foi aprovado para usar o simulador.')
                return redirect(url_for('painel'))
        else:
            flash('Arquivo de usuários não encontrado.')
            return redirect(url_for('login'))

    resultado = error = None
    if request.method == 'POST':
        edo = request.form.get('edo')
        entrada = request.form.get('entrada')
        saida = request.form.get('saida')

        if not edo or not entrada or not saida:
            error = "Preencha todos os campos."
        else:
            try:
                Ls_expr, FT = parse_edo(edo, entrada, saida)
                ft_latex = ft_to_latex(Ls_expr)
                t_open, y_open = resposta_degrau(FT)
                L, T = estima_LT(t_open, y_open)
                Kp, Ki, Kd = sintonia_ziegler_nichols(L, T)
                pid = cria_pid_tf(Kp, Ki, Kd)
                mf = malha_fechada_tf(FT, pid)
                t_closed, y_closed = resposta_degrau(mf)

                def tf_to_sympy_tf(tf):
                    num = tf.num[0][0]
                    den = tf.den[0][0]
                    s = sp.symbols('s')
                    num_poly = sum(coef * s**(len(num)-i-1) for i, coef in enumerate(num))
                    den_poly = sum(coef * s**(len(den)-i-1) for i, coef in enumerate(den))
                    return num_poly / den_poly

                expr_pid = sp.simplify(tf_to_sympy_tf(pid))
                expr_mf = sp.simplify(tf_to_sympy_tf(mf))

                salvar_grafico_resposta(t_open, y_open, 'resposta_malha_aberta', rotacao=180)
                salvar_grafico_resposta(t_closed, y_closed, 'resposta_malha_fechada', rotacao=90, deslocamento=1.0)
                plot_polos_zeros(FT)

                den_coefs = flatten_and_convert(FT.den[0])
                routh_table = tabela_routh(den_coefs)

                resultado = {
                    'ft_latex': ft_latex,
                    'pid_latex': sp.latex(expr_pid, mul_symbol='dot'),
                    'mf_latex': sp.latex(expr_mf, mul_symbol='dot'),
                    'Kp': Kp,
                    'Ki': Ki,
                    'Kd': Kd,
                    'img_resposta_aberta': 'resposta_malha_aberta.png',
                    'img_resposta_fechada': 'resposta_malha_fechada.png',
                    'img_pz': 'polos_zeros.png',
                    'routh_table': routh_table.tolist()
                }

            except Exception as e:
                error = f"Erro no processamento: {str(e)}"
    return render_template('simulador.html', resultado=resultado, error=error)

@app.route('/perfil')
def perfil():
    if 'usuario_logado' not in session:
        flash('Faça login para acessar o perfil.')
        return redirect(url_for('login'))

    email = session['usuario_logado']
    if os.path.exists('usuarios.json'):
        with open('usuarios.json', 'r') as f:
            usuarios = json.load(f)
    else:
        usuarios = {}

    usuario = usuarios.get(email)
    return render_template('perfil.html', usuario=usuario, email=email)

@app.route('/alterar_senha', methods=['GET', 'POST'])
def alterar_senha():
    if 'usuario_logado' not in session:
        flash('Faça login para alterar a senha.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        senha_atual = request.form.get('senha_atual')
        nova_senha = request.form.get('nova_senha')
        confirmar_senha = request.form.get('confirmar_senha')

        email = session['usuario_logado']
        if os.path.exists('usuarios.json'):
            with open('usuarios.json', 'r') as f:
                usuarios = json.load(f)
        else:
            usuarios = {}

        if usuarios[email]['senha'] != senha_atual:
            flash('Senha atual incorreta.')
            return redirect(url_for('alterar_senha'))

        if nova_senha != confirmar_senha:
            flash('Nova senha e confirmação não conferem.')
            return redirect(url_for('alterar_senha'))

        usuarios[email]['senha'] = nova_senha
        with open('usuarios.json', 'w') as f:
            json.dump(usuarios, f, indent=4)

        flash('Senha alterada com sucesso.')
        return redirect(url_for('perfil'))

    return render_template('alterar_senha.html')

@app.route('/funcao_transferencia')
def funcao_transferencia():
    ft_latex = session.get('ft_latex')
    if not ft_latex:
        ft_latex = "Função de Transferência não disponível."
    return render_template('transferencia.html', ft_latex=ft_latex)

# === EXECUÇÃO PRINCIPAL ===
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)
