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

# === FUNÇÕES AUXILIARES ===

def flatten_and_convert(lst):
    """
    Achata uma lista e tenta converter seus elementos para float.
    Levanta um erro se um elemento simbólico não puder ser avaliado.
    """
    result = []
    for c in lst:
        if hasattr(c, 'iter') and not isinstance(c, (str, bytes)):
            result.extend(flatten_and_convert(c))
        else:
            try:
                result.append(float(c))
            except TypeError:
                raise ValueError(f"Coeficiente simbólico '{c}' não pode ser avaliado para um valor numérico. Por favor, forneça valores numéricos para todos os símbolos para gerar gráficos e sintonias.")
            except Exception as e:
                raise Exception(f"Erro convertendo coeficiente: {c} ({e})")
    return result

def pad_coeffs(num_coeffs, den_coeffs):
    """
    Preenche os coeficientes do numerador e denominador com zeros
    para que tenham o mesmo comprimento para a função de transferência.
    """
    len_num = len(num_coeffs)
    len_den = len(den_coeffs)
    if len_num < len_den:
        num_coeffs = [0] * (len_den - len_num) + num_coeffs
    elif len_den < len_num:
        den_coeffs = [0] * (len_num - len_den) + den_coeffs
    return num_coeffs, den_coeffs

def parse_edo(edo_str, entrada_str, saida_str):
    """
    Analisa uma Equação Diferencial Ordinária (EDO) para obter sua Função de Transferência.
    Permite variáveis de entrada e saída flexíveis e lida com coeficientes simbólicos.
    """
    t = sp.symbols('t', real=True)
    
    # Cria funções simbólicas para entrada e saída dinamicamente
    X = sp.Function(saida_str)(t)
    F = sp.Function(entrada_str)(t)

    # Dicionário local para sympify, incluindo as novas funções X e F
    local_dict = {
        'sp': sp, 't': t,
        saida_str: X, entrada_str: F,
    }
    
    # Adiciona as derivadas de X e F ao local_dict para reconhecimento pelo sympify
    # Suporta até 4ª derivada, pode ser ajustado conforme a necessidade
    for i in range(1, 5):
        local_dict[f'diff({saida_str},t,{i})'] = sp.Derivative(X, t, i)
        local_dict[f'diff({entrada_str},t,{i})'] = sp.Derivative(F, t, i)
    # Adiciona também a primeira derivada sem o número explícito (diff(y,t))
    local_dict[f'diff({saida_str},t)'] = sp.Derivative(X, t, 1)
    local_dict[f'diff({entrada_str},t)'] = sp.Derivative(F, t, 1)


    # Prepara a string da EDO para sympify
    eq_str = edo_str.replace('diff', 'sp.Derivative')
    if '=' in eq_str:
        lhs, rhs = eq_str.split('=')
        eq_str = f"({lhs.strip()}) - ({rhs.strip()})"

    # Converte a string da EDO para uma expressão simbólica do SymPy
    try:
        eq = sp.sympify(eq_str, locals=local_dict)
    except Exception as e:
        raise ValueError(f"Erro ao interpretar a EDO. Verifique a sintaxe. Detalhes: {e}")

    # Símbolos da transformada de Laplace para a saída (Xs) e entrada (Fs)
    Xs = sp.symbols(f'{saida_str}s')
    Fs = sp.symbols(f'{entrada_str}s')
    
    expr_laplace = eq
    
    # Substitui as funções do tempo e suas derivadas por suas transformadas de Laplace
    # As condições iniciais são assumidas como zero para obtenção da Função de Transferência.
    for d in expr_laplace.atoms(sp.Derivative):
        ordem = d.derivative_count
        func = d.expr
        if func == X:
            expr_laplace = expr_laplace.subs(d, s**ordem * Xs)
        elif func == F:
            expr_laplace = expr_laplace.subs(d, s**ordem * Fs)
    
    # Substitui as funções originais (não derivadas) por suas transformadas de Laplace
    expr_laplace = expr_laplace.subs({X: Xs, F: Fs})

    # Isola Xs/Fs para obter a Função de Transferência
    try:
        coef_Xs = expr_laplace.coeff(Xs)
        resto = expr_laplace - coef_Xs * Xs
        
        # Tenta resolver explicitamente para Xs em termos de Fs para robustez
        solution_dict = sp.solve(expr_laplace, Xs)
        if solution_dict:
            # Pega a primeira solução se houver múltiplas (comum em sistemas lineares, deve ser única)
            solution = solution_dict[0]
            Ls_expr = sp.simplify(solution.subs(Fs, 1)) # Assume Fs = 1 para a FT
        else:
            # Caso o solve falhe, tenta a abordagem original de isolamento
            if Fs not in resto.free_symbols:
                # Se não houver Fs no termo restante, a FT pode ser inválida ou constante
                raise ValueError("Não foi possível identificar a relação de entrada/saída na EDO. Verifique se a variável de entrada está presente.")
            Ls_expr = -resto / coef_Xs
            Ls_expr = sp.simplify(Ls_expr.subs(Fs, 1))
    except Exception as e:
        raise ValueError(f"Não foi possível isolar a Função de Transferência. Verifique a EDO e as variáveis de entrada/saída. Erro: {e}")

    # Extrai numerador e denominador da FT simbólica
    num, den = sp.fraction(Ls_expr)
    
    # Identifica quaisquer símbolos remanescentes (coeficientes simbólicos)
    simbolos_num = list(num.free_symbols - {s})
    simbolos_den = list(den.free_symbols - {s})
    
    # Se houver símbolos, avalia-os para 1.0 para permitir cálculos numéricos e gráficos
    if simbolos_num or simbolos_den:
        subs_dict = {sym: 1.0 for sym in simbolos_num + simbolos_den}
        num_eval = num.subs(subs_dict)
        den_eval = den.subs(subs_dict)
        # Informa ao usuário sobre a substituição
        flash(f"Aviso: Coeficientes simbólicos '{', '.join(map(str, simbolos_num + simbolos_den))}' foram detectados e substituídos por 1.0 para a geração dos gráficos e parâmetros PID. Para resultados exatos, forneça apenas coeficientes numéricos na EDO ou revise a sintaxe.", 'warning')
    else:
        num_eval = num
        den_eval = den

    # Converte os coeficientes simbólicos (agora avaliados) para floats
    try:
        num_coeffs = [float(c.evalf()) for c in sp.Poly(num_eval, s).all_coeffs()]
        den_coeffs = [float(c.evalf()) for c in sp.Poly(den_eval, s).all_coeffs()]
    except TypeError:
        raise ValueError("Não foi possível converter os coeficientes da Função de Transferência para valores numéricos. Certifique-se de que todos os coeficientes sejam numéricos após a análise da EDO.")

    # Remove coeficientes zero iniciais para evitar FTs improprias (ex: 0s^2 + 2s + 1)
    while len(den_coeffs) > 1 and den_coeffs[0] == 0:
        den_coeffs.pop(0)
        if num_coeffs and num_coeffs[0] == 0: # Garante que o numerador também seja ajustado se necessário
            num_coeffs.pop(0)
        else: # Se o numerador for mais curto, preenche com zero
            num_coeffs = [0] * (len(den_coeffs) - len(num_coeffs)) + num_coeffs
    
    # Preenche os coeficientes para que numerador e denominador tenham o mesmo comprimento
    num_coeffs, den_coeffs = pad_coeffs(num_coeffs, den_coeffs)
    
    if not den_coeffs or all(c == 0 for c in den_coeffs):
        raise ValueError("O denominador da função de transferência resultou em zero ou está vazio. Verifique a equação diferencial.")
    
    # Cria a Função de Transferência usando a biblioteca control
    FT = control.TransferFunction(num_coeffs, den_coeffs)
    return Ls_expr, FT

def ft_to_latex(expr):
    """Converte uma expressão SymPy para uma string LaTeX."""
    return sp.latex(expr, mul_symbol='dot')

def resposta_degrau(FT, tempo=None):
    """Calcula a resposta ao degrau de uma Função de Transferência."""
    if tempo is None:
        tempo = np.linspace(0, 10, 1000)
    try:
        t, y = step(FT, T=tempo)
        return t, y
    except Exception as e:
        raise ValueError(f"Não foi possível calcular a resposta ao degrau. Pode ser devido a polos instáveis ou erro na função de transferência. Erro: {e}")

def estima_LT(t, y):
    """
    Estima os parâmetros de atraso (L) e constante de tempo (T)
    a partir da resposta ao degrau, usando o método de 63%.
    """
    y = np.array(y)
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("A resposta ao degrau contém valores inválidos (NaN/Inf), impossível estimar L e T.")

    y_final = y[-1]
    y_inicial = y[0]

    # Lidar com casos onde a resposta não varia significativamente
    if abs(y_final - y_inicial) < 1e-6:
        return 0.01, 0.01 # Retorna valores pequenos para evitar divisão por zero ou erros

    # Normalizar a resposta entre 0 e 1 (ou -1 e 0 se for decrescente)
    y_scaled = (y - y_inicial) / (y_final - y_inicial)

    try:
        # L - Tempo em que a resposta atinge 1% da variação total
        indice_inicio = next(i for i, v in enumerate(y_scaled) if v > 0.01 or v < -0.01)
        L = t[indice_inicio]
    except StopIteration:
        L = 0.01 # Se não atingir 1%, assume um pequeno atraso

    # T - Tempo em que a resposta atinge 63% da variação total, subtraído de L
    y_63_target = 0.63
    try:
        indice_63 = next(i for i, v in enumerate(y_scaled) if v >= y_63_target)
        T = t[indice_63] - L
    except StopIteration:
        T = 0.01 # Se não atingir 63%, assume um pequeno tempo de atraso

    return (L if L >= 0 else 0.01), (T if T >= 0 else 0.01) # Garante valores positivos e mínimos

def sintonia_ziegler_nichols(L, T):
    """Calcula os parâmetros PID (Kp, Ki, Kd) usando as regras de Ziegler-Nichols."""
    # Evita divisão por zero ou valores muito pequenos que levariam a Kp/Ti muito grandes
    if L == 0: L = 1e-6
    if T == 0: T = 1e-6
    
    Kp = 1.2 * T / L
    Ti = 2 * L
    Td = 0.5 * L
    
    Ki = Kp / Ti if Ti != 0 else 0.0
    Kd = Kp * Td
    return Kp, Ki, Kd

def cria_pid_tf(Kp, Ki, Kd):
    """Cria a Função de Transferência de um controlador PID."""
    s_tf = control.TransferFunction.s # Usar s do control para criar a FT
    return Kp + Ki / s_tf + Kd * s_tf

def malha_fechada_tf(Gp, Gc):
    """Calcula a Função de Transferência de malha fechada."""
    return control.feedback(Gp * Gc, 1)

def tabela_routh(coeficientes):
    """
    Gera a tabela de Routh-Hurwitz para análise de estabilidade.
    """
    # Garante que os coeficientes são numéricos
    coeficientes = [float(c[0]) if isinstance(c, list) else float(c) for c in coeficientes]
    n = len(coeficientes)
    
    if n == 0 or all(c == 0 for c in coeficientes):
        flash("Aviso: Coeficientes do denominador são todos zero ou vazios para a Tabela de Routh.", 'warning')
        return np.array([[]]) # Retorna tabela vazia se não houver coeficientes válidos

    # Remove zeros iniciais se houver (ex: 0s^2 + 2s + 1)
    # Isso é importante para sistemas onde o grau do denominador pode não ser o maior termo
    while n > 0 and abs(coeficientes[0]) < 1e-9: # Usar uma pequena tolerância para zero
        coeficientes.pop(0)
        n = len(coeficientes)
    
    if n == 0:
        flash("Aviso: Todos os coeficientes do denominador se anularam após a remoção de zeros iniciais.", 'warning')
        return np.array([[]])

    # Se apenas um coeficiente restar e for zero, também é inválido
    if n == 1 and abs(coeficientes[0]) < 1e-9:
        flash("Aviso: O único coeficiente do denominador restante é zero.", 'warning')
        return np.array([[]])

    m = (n + 1) // 2
    routh = np.zeros((n, m))
    
    # Preenche as duas primeiras linhas da tabela
    routh[0, :len(coeficientes[::2])] = coeficientes[::2]
    if n > 1:
        routh[1, :len(coeficientes[1::2])] = coeficientes[1::2]
    
    # Calcula as linhas restantes
    for i in range(2, n):
        # Lida com o caso de zero ou quase zero na primeira coluna substituindo por um pequeno epsilon
        if abs(routh[i - 1, 0]) < 1e-9:
            routh[i - 1, 0] = 1e-9 # Usar um valor muito pequeno para evitar divisão por zero
            flash(f"Aviso: Zero ou valor muito próximo de zero detectado na primeira coluna da linha {i-1} da Tabela de Routh. Substituindo por um pequeno valor (1e-9) para continuar o cálculo. Isso pode indicar polos no eixo imaginário ou casos especiais de estabilidade.", 'warning')

        for j in range(m - 1):
            a = routh[i - 2, 0]
            b = routh[i - 1, 0]
            c = routh[i - 2, j + 1]
            d = routh[i - 1, j + 1]
            
            # Evita divisão por zero explícita
            if abs(b) < 1e-9: # Se o divisor for muito pequeno
                routh[i, j] = 0 # Considera como 0 para evitar NaN/Inf
            else:
                routh[i, j] = (b * c - a * d) / b
                
    return routh

def salvar_grafico_resposta(t, y, nome, rotacao=0, deslocamento=0.0):
    """
    Salva um gráfico de resposta ao degrau.
    Permite rotação e deslocamento dos dados.
    """
    t = np.array(t)
    y = np.array(y)

    # Remove NaN e Inf para evitar erros de plotagem
    valid_indices = np.isfinite(t) & np.isfinite(y)
    t = t[valid_indices]
    y = y[valid_indices]

    if len(t) == 0:
        print(f"Aviso: Dados vazios ou inválidos para o gráfico '{nome}'. Não será gerado.")
        return None

    y = y + deslocamento

    # Aplica rotação e deslocamento
    if rotacao == 180:
        t = -t[::-1]
        y = -y[::-1]
    elif rotacao == 90:
        # Troca e inverte eixos para 90 graus
        t, y = y, t # Troca os eixos
        t = -t # Inverte o novo eixo "x" para simular rotação no sentido anti-horário
        y = y - y.min() # Garante que o novo eixo "y" comece de zero ou positivo
    
    # Normaliza o tempo e a saída para que comecem em zero ou positivo
    if t.size > 0 and min(t) < 0:
        t = t - min(t)
    if y.size > 0 and min(y) < 0:
        y = y - min(y)

    plt.figure(figsize=(8, 4))
    plt.plot(t, y, label='Resposta ao Degrau')
    plt.xlabel('Tempo (s)' if rotacao not in [90, 270] else 'Saída') # Ajusta label do eixo x
    plt.ylabel('Saída' if rotacao not in [90, 270] else 'Tempo (s)') # Ajusta label do eixo y
    plt.title(nome)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    caminho = os.path.join('static', f'{nome}.png')
    try:
        plt.savefig(caminho)
    except Exception as e:
        print(f"Erro ao salvar o gráfico '{nome}': {e}")
        caminho = None # Retorna None se não conseguir salvar
    finally:
        plt.close() # Sempre fecha o plot para liberar memória
    return caminho

def plot_polos_zeros(FT):
    """Cria e salva o diagrama de polos e zeros."""
    try:
        poles = FT.poles()
        zeros = FT.zeros()
    except Exception as e:
        print(f"Erro ao obter polos e zeros da FT: {e}")
        flash("Não foi possível determinar os polos e zeros da Função de Transferência.", 'warning')
        return None

    fig, ax = plt.subplots()
    
    # Plota polos (marcadores 'x', vermelhos)
    if poles.size > 0:
        ax.scatter(np.real(poles), np.imag(poles), marker='x', color='red', label='Polos', s=100)
    # Plota zeros (marcadores 'o', azuis)
    if zeros.size > 0:
        ax.scatter(np.real(zeros), np.imag(zeros), marker='o', color='blue', label='Zeros', s=100)
        
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Parte Real')
    ax.set_ylabel('Parte Imaginária')
    ax.set_title('Diagrama de Polos e Zeros')
    ax.legend()
    ax.grid(True)
    
    # Ajusta os limites do gráfico dinamicamente
    all_coords_real = np.concatenate((np.real(poles), np.real(zeros)))
    all_coords_imag = np.concatenate((np.imag(poles), np.imag(zeros)))

    if all_coords_real.size > 0 and all_coords_imag.size > 0:
        min_re, max_re = all_coords_real.min(), all_coords_real.max()
        min_im, max_im = all_coords_imag.min(), all_coords_imag.max()

        # Adiciona uma margem para melhor visualização
        margin_re = max(0.5, (max_re - min_re) * 0.1)
        margin_im = max(0.5, (max_im - min_im) * 0.1)

        ax.set_xlim(min_re - margin_re, max_re + margin_re)
        ax.set_ylim(min_im - margin_im, max_im + margin_im)
    else: # Caso de FT constante (sem polos/zeros finitos)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

    caminho = os.path.join('static', 'polos_zeros.png')
    try:
        plt.savefig(caminho)
    except Exception as e:
        print(f"Erro ao salvar o gráfico de Polos e Zeros: {e}")
        caminho = None
    finally:
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

        # Admin fixo
        if email == 'tisaaceng@gmail.com' and senha == '4839AT81':
            session['usuario_logado'] = email
            session['is_admin'] = True
            flash('Login de administrador realizado com sucesso!', 'success')
            return redirect(url_for('admin'))

        # Usuários comuns
        if email in usuarios and usuarios[email]['senha'] == senha:
            if usuarios[email].get('aprovado', False):
                session['usuario_logado'] = email
                session['is_admin'] = False
                flash('Login realizado com sucesso!', 'success')
                return redirect(url_for('painel'))
            else:
                flash('Seu cadastro ainda não foi aprovado.', 'warning')
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

        flash('Cadastro enviado para aprovação.', 'info')
        return redirect(url_for('login'))
    return render_template('cadastro.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    # Permite acesso somente se estiver logado como admin
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

    # Verifica se o usuário está aprovado (apenas para usuários comuns)
    if email != 'tisaaceng@gmail.com':
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
    is_admin = session.get('is_admin', False)

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
                
                # Resposta ao Degrau (Malha Aberta)
                t_open, y_open = resposta_degrau(FT)
                
                # Estima L e T e calcula parâmetros PID
                L, T = estima_LT(t_open, y_open)
                Kp, Ki, Kd = sintonia_ziegler_nichols(L, T)
                
                # Cria controlador PID e FT de Malha Fechada
                pid = cria_pid_tf(Kp, Ki, Kd)
                mf = malha_fechada_tf(FT, pid)
                
                # Resposta ao Degrau (Malha Fechada)
                t_closed, y_closed = resposta_degrau(mf)

                # Função auxiliar para converter FT de control para SymPy
                def tf_to_sympy_tf(tf_control_obj):
                    num_list = tf_control_obj.num[0][0]
                    den_list = tf_control_obj.den[0][0]

                    num_poly = sum(coef * s**(len(num_list) - i - 1) for i, coef in enumerate(num_list))
                    den_poly = sum(coef * s**(len(den_list) - i - 1) for i, coef in enumerate(den_list))
                    return num_poly / den_poly

                # Converte FTs para SymPy para exibição em LaTeX
                expr_pid = sp.simplify(tf_to_sympy_tf(pid))
                expr_mf = sp.simplify(tf_to_sympy_tf(mf))

                # Salva os gráficos
                img_resposta_aberta = salvar_grafico_resposta(t_open, y_open, 'resposta_malha_aberta', rotacao=0)
                img_resposta_fechada = salvar_grafico_resposta(t_closed, y_closed, 'resposta_malha_fechada', rotacao=0)
                img_pz = plot_polos_zeros(FT)

                # Coeficientes do denominador para a Tabela de Routh
                den_coeffs = flatten_and_convert(FT.den[0])
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

            except ValueError as ve:
                error = f"Erro de validação ou cálculo: {str(ve)}"
            except Exception as e:
                error = f"Erro inesperado no processamento: {str(e)}. Por favor, verifique a sintaxe da EDO ou os dados de entrada."
    
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

        flash('Senha alterada com sucesso!', 'success')
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


# === EXECUÇÃO PRINCIPAL ===

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)
