from flask import Flask, render_template, request, redirect, url_for, flash, session
import json
import os
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import control
from control.matlab import step
from sympy.abc import s # s já importado, mas reforça que é para a transformada de Laplace

app = Flask(__name__)
app.secret_key = 'sua_chave_secreta'

=== FUNÇÕES AUXILIARES ===

def flatten_and_convert(lst):
    result = []
    for c in lst:
        if hasattr(c, 'iter') and not isinstance(c, (str, bytes)):
            result.extend(flatten_and_convert(c))
        else:
            try:
                # Tenta converter para float. Se for simbólico e não puder ser avaliado, lança erro
                result.append(float(c))
            except TypeError: # Captura erro se o símbolo não puder ser avaliado para float
                raise ValueError(f"Coeficiente simbólico '{c}' não pode ser avaliado para um valor numérico. Por favor, forneça valores numéricos para todos os símbolos para gerar gráficos e sintonias.")
            except Exception as e:
                raise Exception(f"Erro convertendo coeficiente: {c} ({e})")
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
    # Criar funções simbólicas para entrada e saída dinamicamente
    X = sp.Function(saida_str)(t)
    F = sp.Function(entrada_str)(t)

    # Dicionário local para sympify, incluindo as novas funções X e F
    # e quaisquer outras derivadas que possam aparecer
    local_dict = {
        'sp': sp, 't': t,
        saida_str: X, entrada_str: F,
    }
    
    # Adiciona as derivadas de X e F ao local_dict
    # Isso ajuda o sympify a reconhecer 'diff(y,t)' como sp.Derivative(y(t), t)
    for i in range(1, 5): # Suporta até 4ª derivada, pode ser ajustado
        local_dict[f'diff({saida_str},t,{i})'] = sp.Derivative(X, t, i)
        local_dict[f'diff({entrada_str},t,{i})'] = sp.Derivative(F, t, i)
        local_dict[f'diff({saida_str},t)'] = sp.Derivative(X, t) # Para primeira derivada
        local_dict[f'diff({entrada_str},t)'] = sp.Derivative(F, t) # Para primeira derivada


    eq_str = edo_str.replace('diff', 'sp.Derivative')
    if '=' in eq_str:
        lhs, rhs = eq_str.split('=')
        eq_str = f"({lhs.strip()}) - ({rhs.strip()})"

    eq = sp.sympify(eq_str, locals=local_dict)

    # Símbolos da transformada de Laplace para a saída (Xs) e entrada (Fs)
    Xs = sp.symbols(f'{saida_str}s')
    Fs = sp.symbols(f'{entrada_str}s')
    
    expr_laplace = eq
    # Substituir as funções e suas derivadas por suas transformadas de Laplace
    # As condições iniciais são assumidas como zero para FT.
    for d in expr_laplace.atoms(sp.Derivative):
        ordem = d.derivative_count
        func = d.expr
        if func == X:
            expr_laplace = expr_laplace.subs(d, s**ordem * Xs)
        elif func == F:
            expr_laplace = expr_laplace.subs(d, s**ordem * Fs)
    
    # Substituir as funções originais por suas transformadas de Laplace (se não forem derivadas)
    expr_laplace = expr_laplace.subs({X: Xs, F: Fs})

    # Isolar Xs/Fs
    try:
        coef_Xs = expr_laplace.coeff(Xs)
        resto = expr_laplace - coef_Xs * Xs
        
        # Certifica-se de que Fs está presente no resto para formar a FT
        if Fs not in resto.free_symbols:
            # Isso pode acontecer se a EDO não tiver um termo de entrada (Fs)
            # ou se a EDO não for homogênea e o Sympy não conseguir isolar Fs corretamente.
            # Tenta resolver explicitamente para Xs em termos de Fs
            solution = sp.solve(expr_laplace, Xs)[0]
            Ls_expr = sp.simplify(solution.subs(Fs, 1)) # Assume Fs = 1 para a FT
        else:
            Ls_expr = -resto / coef_Xs
            Ls_expr = sp.simplify(Ls_expr.subs(Fs, 1))
    except Exception as e:
        raise ValueError(f"Não foi possível isolar a Função de Transferência. Verifique a EDO e as variáveis de entrada/saída. Erro: {e}")

    num, den = sp.fraction(Ls_expr)
    
    # Extrair coeficientes e tentar convertê-los para float.
    # Se houver símbolos, haverá um erro aqui, que será capturado pela função flatten_and_convert.
    # A FT simbólica pode ser gerada, mas a numérica (para gráficos) não.
    
    # Substituir quaisquer símbolos remanescentes no numerador e denominador por 1 para gerar a FT numérica.
    # Isso é um fallback para permitir que os gráficos sejam gerados, mas pode não ser a representação desejada.
    # Um aviso ao usuário seria ideal aqui.
    simbolos_num = list(num.free_symbols - {s})
    simbolos_den = list(den.free_symbols - {s})
    
    if simbolos_num or simbolos_den:
        # Tenta uma substituição padrão para permitir a continuação
        subs_dict = {sym: 1.0 for sym in simbolos_num + simbolos_den}
        num_eval = num.subs(subs_dict)
        den_eval = den.subs(subs_dict)
        flash(f"Aviso: Coeficientes simbólicos '{', '.join(map(str, simbolos_num + simbolos_den))}' foram detectados e substituídos por 1.0 para a geração dos gráficos e parâmetros PID. Para resultados exatos, forneça apenas coeficientes numéricos na EDO ou revise a sintaxe.", 'warning')
    else:
        num_eval = num
        den_eval = den

    try:
        num_coeffs = [float(c.evalf()) for c in sp.Poly(num_eval, s).all_coeffs()]
        den_coeffs = [float(c.evalf()) for c in sp.Poly(den_eval, s).all_coeffs()]
    except TypeError: # Se evalf() não conseguir avaliar devido a símbolos não substituídos
        raise ValueError("Não foi possível converter os coeficientes da Função de Transferência para valores numéricos. Certifique-se de que todos os coeficientes sejam numéricos após a análise da EDO.")


    num_coeffs, den_coeffs = pad_coeffs(num_coeffs, den_coeffs)
    
    if not den_coeffs or all(c == 0 for c in den_coeffs):
        raise ValueError("O denominador da função de transferência resultou em zero ou está vazio. Verifique a equação diferencial.")
        
    # Verifica se o primeiro coeficiente do denominador é zero, o que pode causar problemas
    if den_coeffs[0] == 0 and len(den_coeffs) > 1:
        # Se o termo de maior ordem for zero, tenta remover zeros iniciais
        # Isso pode acontecer se a FT for impropria ou mal formada por algum motivo
        while len(den_coeffs) > 1 and den_coeffs[0] == 0:
            den_coeffs.pop(0)
            num_coeffs.pop(0) # Remove correspondente do numerador para manter o alinhamento
        if den_coeffs[0] == 0: # Se ainda for zero após remover zeros, é um problema
             raise ValueError("O denominador da função de transferência é inválido após simplificação.")


    FT = control.TransferFunction(num_coeffs, den_coeffs)
    return Ls_expr, FT

def ft_to_latex(expr):
    return sp.latex(expr, mul_symbol='dot')

def resposta_degrau(FT, tempo=None):
    if tempo is None:
        tempo = np.linspace(0, 10, 1000)
    try:
        t, y = step(FT, T=tempo)
        return t, y
    except Exception as e:
        raise ValueError(f"Não foi possível calcular a resposta ao degrau. Pode ser devido a polos instáveis ou erro na função de transferência. Erro: {e}")

def estima_LT(t, y):
    # Assegurar que y seja numérico e não contenha NaN ou inf
    y = np.array(y)
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("A resposta ao degrau contém valores inválidos (NaN/Inf), impossível estimar L e T.")

    y_final = y[-1]
    # Lidar com casos onde a resposta não sobe (ex: sistema instável ou constante zero)
    if abs(y_final) < 1e-6: # Se a resposta final é essencialmente zero
        return 0.01, 0.01 # Retorna valores pequenos para evitar divisão por zero ou erros

    # Evita erros para sistemas que começam em zero e vão para zero
    if max(y) - min(y) < 1e-6:
        return 0.01, 0.01

    # Normalizar para 0-100% para L e T
    y_scaled = (y - y[0]) / (y_final - y[0])

    try:
        indice_inicio = next(i for i, v in enumerate(y_scaled) if v > 0.01) # 1% do valor final
        L = t[indice_inicio]
    except StopIteration:
        L = 0.01 # Se não atingir 1%, assume um pequeno atraso

    y_63 = 0.63 # 63% do valor final (em escala 0-1)
    try:
        indice_63 = next(i for i, v in enumerate(y_scaled) if v >= y_63)
        T = t[indice_63] - L
    except StopIteration:
        T = 0.01 # Se não atingir 63%, assume um pequeno tempo de atraso

    return (L if L >= 0 else 0.01), (T if T >= 0 else 0.01)


def sintonia_ziegler_nichols(L, T):
    # Evita divisão por zero
    if L == 0: L = 1e-6
    if T == 0: T = 1e-6
    Kp = 1.2 * T / L
    Ti = 2 * L
    Td = 0.5 * L
    Ki = Kp / Ti if Ti != 0 else 0.0
    Kd = Kp * Td
    return Kp, Ki, Kd

def cria_pid_tf(Kp, Ki, Kd):
    s = control.TransferFunction.s
    # Garante que Ki/s não cause divisão por zero simbólica se s for zero (control.TransferFunction lida com isso numericamente)
    # Usa um pequeno epsilon para evitar erros se Kp, Ki, Kd forem zero, mas control.TransferFunction já cuida disso.
    return Kp + Ki / s + Kd * s

def malha_fechada_tf(Gp, Gc):
    return control.feedback(Gp * Gc, 1)

def tabela_routh(coeficientes):
    # Certifica que os coeficientes são numéricos antes de construir a tabela
    coeficientes = [float(c[0]) if isinstance(c, list) else float(c) for c in coeficientes]
    n = len(coeficientes)
    
    if n == 0 or all(c == 0 for c in coeficientes):
        return np.array([[]]) # Retorna tabela vazia ou com zeros se não houver coeficientes válidos

    # Remove zeros iniciais se houver (ex: 0s^2 + 2s + 1)
    while n > 0 and coeficientes[0] == 0:
        coeficientes.pop(0)
        n = len(coeficientes)
    
    if n == 0:
        return np.array([[]]) # Retorna tabela vazia se tudo for zero

    m = (n + 1) // 2
    routh = np.zeros((n, m))
    
    # Preenche as duas primeiras linhas
    routh[0, :len(coeficientes[::2])] = coeficientes[::2]
    if n > 1:
        routh[1, :len(coeficientes[1::2])] = coeficientes[1::2]
    
    # Calcula as linhas restantes
    for i in range(2, n):
        # Verifica se a primeira coluna da linha anterior é zero
        if routh[i - 1, 0] == 0:
            # Lida com o caso de zero na primeira coluna substituindo por um pequeno epsilon
            routh[i - 1, 0] = 1e-9 # Usar um valor muito pequeno
            flash(f"Aviso: Zero detectado na primeira coluna da linha {i-1} da Tabela de Routh. Substituindo por um pequeno valor (1e-9) para continuar o cálculo. Isso pode indicar polos no eixo imaginário ou problemas de estabilidade.", 'warning')

        for j in range(m - 1):
            a = routh[i - 2, 0]
            b = routh[i - 1, 0]
            c = routh[i - 2, j + 1]
            d = routh[i - 1, j + 1]
            
            # Evita divisão por zero explícita
            if b == 0:
                routh[i, j] = 0 # Ou algum valor indicando indefinição
            else:
                routh[i, j] = (b * c - a * d) / b
                
    return routh

def salvar_grafico_resposta(t, y, nome, rotacao=0, deslocamento=0.0):
    # Certifica que t e y são arrays numpy para operações
    t = np.array(t)
    y = np.array(y)

    # Remove NaN e Inf para evitar erros de plotagem
    valid_indices = np.isfinite(t) & np.isfinite(y)
    t = t[valid_indices]
    y = y[valid_indices]

    if len(t) == 0:
        print(f"Aviso: Dados vazios para o gráfico '{nome}'. Não será gerado.")
        return None

    y = y + deslocamento

    # Aplica rotação e deslocamento conforme especificado
    if rotacao == 180:
        t = -t[::-1]
        y = -y[::-1]
    elif rotacao == 90:
        t, y = -y, t

    # Normaliza o tempo e a saída para que comecem em zero ou positivo, se necessário para visualização
    if t.size > 0 and min(t) < 0:
        t = t - min(t)
    if y.size > 0 and min(y) < 0:
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
    # Verifica se a FT possui polos e zeros definidos.
    # Em caso de FT trivial (ex: constante), poles() e zeros() podem retornar arrays vazios.
    try:
        poles = FT.poles()
        zeros = FT.zeros()
    except Exception as e:
        print(f"Erro ao obter polos e zeros: {e}")
        return None # Retorna None se não conseguir obter polos/zeros

    fig, ax = plt.subplots()
    if poles.size > 0:
        ax.scatter(np.real(poles), np.imag(poles), marker='x', color='red', label='Polos')
    if zeros.size > 0:
        ax.scatter(np.real(zeros), np.imag(zeros), marker='o', color='blue', label='Zeros')
        
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_title('Diagrama de Polos e Zeros')
    ax.legend()
    ax.grid(True)
    
    # Ajusta os limites do gráfico se não houver polos/zeros visíveis, ou se estiverem muito próximos de zero
    if poles.size > 0 or zeros.size > 0:
        all_coords_real = np.concatenate((np.real(poles), np.real(zeros)))
        all_coords_imag = np.concatenate((np.imag(poles), np.imag(zeros)))
        
        if all_coords_real.size > 0 and all_coords_imag.size > 0:
            min_re, max_re = all_coords_real.min(), all_coords_real.max()
            min_im, max_im = all_coords_imag.min(), all_coords_imag.max()

            # Adiciona uma margem aos limites
            margin_re = max(0.5, (max_re - min_re) * 0.1)
            margin_im = max(0.5, (max_im - min_im) * 0.1)

            ax.set_xlim(min_re - margin_re, max_re + margin_re)
            ax.set_ylim(min_im - margin_im, max_im + margin_im)
        else: # Caso de FT constante, por exemplo, onde não há polos nem zeros
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)

    caminho = os.path.join('static', 'polos_zeros.png')
    plt.savefig(caminho)
    plt.close()
    return caminho


=== ROTAS ===

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
                flash('Cadastro ainda não aprovado.', 'warning')
                return redirect(url_for('login'))
        else:
            flash('Credenciais inválidas.', 'danger')
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
            flash('Email já cadastrado ou reservado.', 'warning')
            return redirect(url_for('cadastro'))

        usuarios[email] = {'nome': nome, 'senha': senha, 'aprovado': False}
        with open('usuarios.json', 'w') as f:
            json.dump(usuarios, f, indent=4)

        flash('Cadastro enviado para aprovação.', 'success')
        return redirect(url_for('login'))
    return render_template('cadastro.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    # Permite acesso somente se estiver logado como admin (email fixo + flag)
    if 'usuario_logado' not in session or session.get('is_admin') != True:
        flash('Acesso negado. Apenas o administrador pode acessar.', 'danger')
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
                flash(f'{email} aprovado com sucesso!', 'success')
            else:
                flash('Usuário não encontrado.', 'danger')
        else:
            flash('Arquivo de usuários não encontrado.', 'warning')

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
        flash('Acesso negado.', 'danger')
        return redirect(url_for('login'))
    # Passa a informação de admin para o painel para controlar a visibilidade do link 'Admin'
    is_admin = session.get('is_admin', False)
    return render_template('painel.html', admin=is_admin)

@app.route('/logout')
def logout():
    session.pop('usuario_logado', None)
    session.pop('is_admin', None)
    flash('Logout realizado.', 'info')
    return redirect(url_for('login'))

@app.route('/simulador', methods=['GET', 'POST'])
def simulador():
    if 'usuario_logado' not in session:
        flash('Faça login para acessar o simulador.', 'warning')
        return redirect(url_for('login'))

    email = session['usuario_logado']

    # Verificar se usuário está aprovado
    if email != 'tisaaceng@gmail.com':  # Admin não precisa estar aprovado para acessar o simulador
        if os.path.exists('usuarios.json'):
            with open('usuarios.json', 'r') as f:
                usuarios = json.load(f)
            if not usuarios.get(email, {}).get('aprovado', False):
                flash('Seu cadastro ainda não foi aprovado para usar o simulador.', 'warning')
                return redirect(url_for('painel'))
        else:
            flash('Arquivo de usuários não encontrado.', 'warning')
            return redirect(url_for('login'))

    resultado = error = None
    is_admin = session.get('is_admin', False) # Adicionado para passar para o template

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
                
                # Gera as imagens independentemente se a FT é simbólica ou numérica (com valores padrão)
                t_open, y_open = resposta_degrau(FT)
                L, T = estima_LT(t_open, y_open)
                Kp, Ki, Kd = sintonia_ziegler_nichols(L, T)
                pid = cria_pid_tf(Kp, Ki, Kd)
                mf = malha_fechada_tf(FT, pid)
                t_closed, y_closed = resposta_degrau(mf)

                def tf_to_sympy_tf(tf_control):
                    # Garante que tf_control.num[0][0] e tf_control.den[0][0] são listas
                    num_list = tf_control.num[0][0]
                    den_list = tf_control.den[0][0]

                    # Converte os coeficientes numéricos de volta para SymPy Poly
                    num_poly = sum(coef * s**(len(num_list) - i - 1) for i, coef in enumerate(num_list))
                    den_poly = sum(coef * s**(len(den_list) - i - 1) for i, coef in enumerate(den_list))
                    return num_poly / den_poly

                expr_pid = sp.simplify(tf_to_sympy_tf(pid))
                expr_mf = sp.simplify(tf_to_sympy_tf(mf))

                # Salvar gráficos. Verifica se o retorno não é None (em caso de erro ou dados vazios)
                img_resposta_aberta = salvar_grafico_resposta(t_open, y_open, 'resposta_malha_aberta', rotacao=180)
                img_resposta_fechada = salvar_grafico_resposta(t_closed, y_closed, 'resposta_malha_fechada', rotacao=90, deslocamento=1.0)
                img_pz = plot_polos_zeros(FT)

                # Coeficientes do denominador para a Tabela de Routh (usando a FT numérica)
                den_coeffs = flatten_and_convert(FT.den[0]) # Já devem ser numéricos
                routh_table = tabela_routh(den_coeffs)


                resultado = {
                    'ft_latex': ft_latex,
                    'pid_latex': sp.latex(expr_pid, mul_symbol='dot'),
                    'mf_latex': sp.latex(expr_mf, mul_symbol='dot'),
                    'Kp': Kp,
                    'Ki': Ki,
                    'Kd': Kd,
                    'img_resposta_aberta': img_resposta_aberta,
                    'img_resposta_fechada': img_resposta_fechada,
                    'img_pz': img_pz,
                    'routh_table': routh_table.tolist()
                }

            except ValueError as ve: # Captura erros específicos de validação ou conversão
                error = f"Erro de validação ou cálculo: {str(ve)}"
            except Exception as e:
                error = f"Erro inesperado no processamento: {str(e)}. Por favor, verifique a sintaxe da EDO."
    
    # Passa o status de admin para o template
    return render_template('simulador.html', resultado=resultado, error=error, admin=is_admin)

@app.route('/perfil')
def perfil():
    if 'usuario_logado' not in session:
        flash('Faça login para acessar o perfil.', 'warning')
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
        flash('Faça login para alterar a senha.', 'warning')
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

        if email not in usuarios:
            flash('Usuário não encontrado.', 'danger')
            return redirect(url_for('alterar_senha'))

        if usuarios[email]['senha'] != senha_atual:
            flash('Senha atual incorreta.', 'danger')
            return redirect(url_for('alterar_senha'))

        if nova_senha != confirmar_senha:
            flash('Nova senha e confirmação não conferem.', 'danger')
            return redirect(url_for('alterar_senha'))
        
        if not nova_senha:
            flash('A nova senha não pode ser vazia.', 'danger')
            return redirect(url_for('alterar_senha'))


        usuarios[email]['senha'] = nova_senha
        with open('usuarios.json', 'w') as f:
            json.dump(usuarios, f, indent=4)

        flash('Senha alterada com sucesso.', 'success')
        return redirect(url_for('perfil'))

    return render_template('alterar_senha.html')

@app.route('/funcao_transferencia')
def funcao_transferencia():
    # Esta rota pode não ser mais necessária se a FT já é exibida no simulador
    # Mas a mantive para preservar as rotas existentes
    ft_latex = session.get('ft_latex') # Verifica se a FT foi armazenada na sessão
    if not ft_latex:
        ft_latex = "Função de Transferência não disponível ou não calculada recentemente."
    return render_template('transferencia.html', ft_latex=ft_latex)


=== EXECUÇÃO PRINCIPAL ===

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)
