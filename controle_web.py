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
# !!! AVISO DE SEGURANÇA CRÍTICO !!!
# Esta chave secreta deve ser um valor longo, aleatório e mantida em segredo absoluto.
# Em produção, use uma variável de ambiente (ex: os.environ.get('SECRET_KEY')).
# Nunca a deixe hardcoded desta forma.
app.secret_key = 'sua_chave_secreta_MUITO_SECRETA_E_ALEATORIA_PRODUCAO' # Mantenha esta chave secreta e única em produção

# === FUNÇÕES AUXILIARES ===

def flatten_and_convert(lst):
    result = []
    for c in lst:
        if hasattr(c, '__iter__') and not isinstance(c, (str, bytes)):
            result.extend(flatten_and_convert(c))
        else:
            try:
                result.append(float(c))
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
    # Define as funções simbólicas x(t) e F(t)
    x = sp.Function(saida_str)(t) # x é agora o objeto função x(t)
    F = sp.Function(entrada_str)(t) # F é agora o objeto função F(t)

    # Não substitua 'diff' aqui. Confie no local_dict para mapear as strings 'diff(...)'
    eq_str_to_sympify = edo_str
    if '=' in eq_str_to_sympify:
        lhs, rhs = eq_str_to_sympify.split('=')
        eq_str_to_sympify = f"({lhs.strip()}) - ({rhs.strip()})"

    # local_dict: Mapeia símbolos e representações de string para objetos SymPy
    local_dict = {
        'sp': sp, 't': t,
        # Mapeia os nomes das variáveis para seus objetos de função SymPy
        entrada_str: F, # Ex: 'F': F(t)
        saida_str: x    # Ex: 'x': x(t)
    }
    
    # Explicitamente mapeia as formas de string das derivadas para os objetos SymPy Derivative
    # Isso é crucial para que sympify interprete corretamente 'diff(x,t,2)' etc.
    for i in range(1, 5): # Suporta até a 4ª derivada
        # Ex: 'diff(x,t,2)': sp.Derivative(x(t), t, t)
        local_dict[f'diff({saida_str},t,{i})'] = sp.Derivative(x, t, i)
        local_dict[f'diff({entrada_str},t,{i})'] = sp.Derivative(F, t, i)
    # Também para as primeiras derivadas sem o '1' explícito
    local_dict[f'diff({saida_str},t)'] = sp.Derivative(x, t, 1)
    local_dict[f'diff({entrada_str},t)'] = sp.Derivative(F, t, 1)

    try:
        # Passa a string da EDO original diretamente para sympify, permitindo que local_dict manipule 'diff'
        eq = sp.sympify(eq_str_to_sympify, locals=local_dict)
    except Exception as e:
        raise ValueError(f"Erro ao interpretar a Equação Diferencial Ordinária (EDO): '{e}'. Verifique a sintaxe da EDO. Ex: 'diff(x,t,2) + x = F'")

    # Definir Xs e Fs com base nas strings de entrada/saída
    Xs = sp.symbols(f'{saida_str}s')
    Fs = sp.symbols(f'{entrada_str}s')

    # O mapa de substituição precisa encontrar os *objetos SymPy reais* criados por sympify
    expr_laplace = eq
    subs_map = {}
    for atom in expr_laplace.atoms():
        if isinstance(atom, sp.Function):
            # Compara diretamente os objetos de função SymPy
            if atom == x: 
                subs_map[atom] = Xs
            elif atom == F: 
                subs_map[atom] = Fs
        elif isinstance(atom, sp.Derivative):
            base_func = atom.expr
            order = atom.derivative_count
            # Compara diretamente o objeto da função base da derivada
            if base_func == x: 
                subs_map[atom] = s**order * Xs
            elif base_func == F: 
                subs_map[atom] = s**order * Fs
            
    expr_laplace = expr_laplace.subs(subs_map)

    # Verificar se restaram termos no domínio do tempo após a substituição
    for atom in expr_laplace.atoms():
        if (isinstance(atom, sp.Function) and atom.args == (t,)) or \
           (isinstance(atom, sp.Derivative) and atom.expr.args == (t,)):
            raise ValueError(f"Erro na transformação de Laplace: A equação ainda contém termos no domínio do tempo como '{atom}'. Certifique-se de que todas as funções de tempo e suas derivadas foram corretamente especificadas ou substituídas.")

    # === INÍCIO DAS MELHORIAS NAS MENSAGENS DE ERRO (já existentes) ===
    try:
        coef_Xs = expr_laplace.coeff(Xs)
        
        if coef_Xs == 0:
            if Xs not in expr_laplace.free_symbols:
                raise ValueError(f"A variável de saída '{saida_str}' (transformada em '{Xs}') não foi encontrada na equação após a transformação de Laplace. Verifique se a EDO realmente depende da variável de saída fornecida ou se a sintaxe está correta (ex: 'x' em vez de 'X').")
            else:
                raise ValueError(f"Os termos da variável de saída '{saida_str}' (transformada em '{Xs}') na EDO se cancelaram ou resultaram em um coeficiente nulo ({coef_Xs}) após a transformação de Laplace. Isso impede o cálculo da Função de Transferência. Verifique a linearidade ou a forma da EDO.")

        resto = expr_laplace - coef_Xs * Xs
        
        if Fs not in resto.free_symbols and resto != 0:
            raise ValueError(f"Não foi possível isolar a Função de Transferência. Termos remanescentes sem a variável de entrada '{entrada_str}' (transformada em '{Fs}'): '{resto}'. Verifique se a EDO é linear em F e se a variável de entrada está presente e correta (Ex: use 'k*F' no lugar de 'F*k' se 'k' for um símbolo).")
            
        if resto == 0 and Fs not in expr_laplace.free_symbols:
             raise ValueError(f"Não foi possível identificar a variável de entrada '{entrada_str}' (transformada em '{Fs}') na equação transformada para Laplace. Verifique se a EDO inclui a variável de entrada e se o seu coeficiente não é zero (Ex: 'k*F' em vez de '0*F').")

        Ls_expr = -resto / coef_Xs
        Ls_expr = sp.simplify(Ls_expr.subs(Fs, 1)) # Assume Fs=1 para a FT

    except ValueError as ve: 
        raise ve
    except Exception as e:
        raise ValueError(f"Um erro inesperado ocorreu durante o isolamento da Função de Transferência: {e}. Verifique a EDO e as variáveis de entrada/saída.")
    # === FIM DAS MELHORIAS NAS MENSAGENS DE ERRO ===

    num, den = sp.fraction(Ls_expr)
    
    # Lidar com coeficientes simbólicos
    simbolos_num = list(num.free_symbols - {s})
    simbolos_den = list(den.free_symbols - {s})
    
    if simbolos_num or simbolos_den:
        subs_dict = {sym: 1.0 for sym in simbolos_num + simbolos_den}
        num_eval = num.subs(subs_dict)
        den_eval = den.subs(subs_dict)
        flash(f"Aviso: Coeficientes simbólicos '{', '.join(map(str, simbolos_num + simbolos_den))}' foram detectados e substituídos por 1.0 para a geração dos gráficos e parâmetros PID. Para resultados exatos, forneça apenas coeficientes numéricos na EDO ou revise a sintaxe.", 'warning')
    else:
        num_eval = num
        den_eval = den

    num_coeffs = [float(c.evalf()) for c in sp.Poly(num_eval, s).all_coeffs()]
    den_coeffs = [float(c.evalf()) for c in sp.Poly(den_eval, s).all_coeffs()]
    num_coeffs, den_coeffs = pad_coeffs(num_coeffs, den_coeffs)

    if not den_coeffs or all(c == 0 for c in den_coeffs):
        raise ValueError("O denominador da função de transferência resultou em zero ou está vazio. Verifique a equação diferencial.")
        
    # Remover zeros iniciais do denominador se houver (ex: [0, 1, 2] -> [1, 2])
    while len(den_coeffs) > 1 and abs(den_coeffs[0]) < 1e-9:
        den_coeffs.pop(0)
        # Ajusta o numerador também se a ordem for reduzida
        if num_coeffs and len(num_coeffs) > 0 and abs(num_coeffs[0]) < 1e-9:
            num_coeffs.pop(0)
        else: # Se o numerador é menor, preenche com zeros para manter alinhamento
            num_coeffs = [0] * (len(den_coeffs) - len(num_coeffs)) + num_coeffs


    FT = control.TransferFunction(num_coeffs, den_coeffs)
    return Ls_expr, FT

def ft_to_latex(expr):
    return sp.latex(expr, mul_symbol='dot')

def resposta_degrau(FT, tempo=None):
    if tempo is None:
        tempo = np.linspace(0, 10, 1000)
    try:
        # Aumentar o tempo de simulação para sistemas instáveis ou lentos
        if FT.poles() is not None and np.any(np.real(FT.poles()) >= 0):
            if tempo[-1] < 20: 
                tempo = np.linspace(0, 20, 2000)
        
        t, y = step(FT, T=tempo)
        
        if np.any(np.abs(y) > 1e10): 
            flash("Aviso: A resposta ao degrau apresentou valores muito grandes, indicando um sistema instável. O gráfico pode ser difícil de interpretar.", 'warning')
            y[np.abs(y) > 1e10] = np.sign(y[np.abs(y) > 1e10]) * 1e10

        return t, y
    except Exception as e:
        raise ValueError(f"Não foi possível calcular a resposta ao degrau. Pode ser devido a polos instáveis ou erro na função de transferência. Erro: {e}")

def estima_LT(t, y):
    y_final = y[-1]
    if abs(y_final) < 1e-6 or (max(y) - min(y) < 1e-6): 
        return 0.01, 0.01 

    if abs(y_final) < 1e-6:
        L_threshold = 0.01 * (max(y) - min(y)) + min(y)
        y_63_target = 0.63 * (max(y) - min(y)) + min(y)
    else:
        L_threshold = 0.01 * y_final
        y_63_target = 0.63 * y_final


    indice_inicio = next((i for i, v in enumerate(y) if v > L_threshold), 0)
    L = t[indice_inicio]

    indice_63 = next((i for i, v in enumerate(y) if v >= y_63_target), len(y) - 1)
    T_estimado = t[indice_63] - L

    L = L if L >= 0 else 0.01
    T_estimado = T_estimado if T_estimado >= 0 else 0.01

    return L, T_estimado


def sintonia_ziegler_nichols(L, T):
    Kp = 1.2 * T / L if L != 0 else 1.0
    Ti = 2 * L
    Td = 0.5 * L
    Ki = Kp / Ti if Ti != 0 else 0.0
    Kd = Kp * Td
    return Kp, Ki, Kd

def cria_pid_tf(Kp, Ki, Kd):
    s_sym = control.TransferFunction.s 
    return Kp + Ki / s_sym + Kd * s_sym

def malha_fechada_tf(Gp, Gc):
    return control.feedback(Gp * Gc, 1)

def tabela_routh(coeficientes):
    coeficientes_numericos = [float(c[0]) if isinstance(c, list) else float(c) for c in coeficientes]
    n = len(coeficientes_numericos)
    
    if n == 0 or all(c == 0 for c in coeficientes_numericos):
        flash("Aviso: Coeficientes do denominador são todos zero ou vazios para a Tabela de Routh.", 'warning')
        return np.array([[]])

    while n > 0 and abs(coeficientes_numericos[0]) < 1e-9:
        coeficientes_numericos.pop(0)
        n = len(coeficientes_numericos)
        
    if n == 0:
        flash("Aviso: Todos os coeficientes do denominador se anularam após a remoção de zeros iniciais na Tabela de Routh.", 'warning')
        return np.array([[]])

    if n == 1 and abs(coeficientes_numericos[0]) < 1e-9:
        flash("Aviso: O único coeficiente do denominador restante é zero para a Tabela de Routh.", 'warning')
        return np.array([[]])

    m = (n + 1) // 2
    routh = np.zeros((n, m))
    
    routh[0, :len(coeficientes_numericos[::2])] = coeficientes_numericos[::2]
    if n > 1:
        routh[1, :len(coeficientes_numericos[1::2])] = coeficientes_numericos[1::2]
    
    for i in range(2, n):
        if abs(routh[i - 1, 0]) < 1e-9:
            routh[i - 1, 0] = 1e-9 
            flash(f"Aviso: Zero ou valor muito próximo de zero detectado na primeira coluna da linha {i-1} da Tabela de Routh. Substituindo por um pequeno valor (1e-9) para continuar o cálculo. Isso pode indicar polos no eixo imaginário ou problemas de estabilidade.", 'warning')

        for j in range(m - 1):
            a = routh[i - 2, 0]
            b = routh[i - 1, 0]
            c = routh[i - 2, j + 1]
            d = routh[i - 1, j + 1]
            
            if abs(b) < 1e-9:
                routh[i, j] = 0
            else:
                routh[i, j] = (b * c - a * d) / b
                
    return routh

def salvar_grafico_resposta(t, y, nome, rotacao=0, deslocamento=0.0):
    t = np.array(t)
    y = np.array(y)

    valid_indices = np.isfinite(t) & np.isfinite(y)
    t = t[valid_indices]
    y = y[valid_indices]

    if len(t) == 0:
        print(f"Aviso: Dados vazios ou inválidos para o gráfico '{nome}'. Não será gerado.")
        return None

    y = y + deslocamento

    if rotacao == 180:
        t = -t[::-1]
        y = -y[::-1]
    elif rotacao == 90:
        t, y = -y, t 

    if t.size > 0:
        t_min = t.min()
        if t_min < 0:
            t = t - t_min 
    
    if y.size > 0:
        y_min = y.min()
        if y_min < 0:
            y = y - y_min 

    plt.figure(figsize=(8, 4))
    plt.plot(t, y, label='Resposta ao Degrau')
    plt.xlabel('Tempo (s)' if rotacao != 90 else 'Saída')
    plt.ylabel('Saída' if rotacao != 90 else 'Tempo (s)')
    plt.title(nome)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    caminho = os.path.join('static', f'{nome}.png')
    try:
        os.makedirs('static', exist_ok=True)
        plt.savefig(caminho)
    except Exception as e:
        print(f"Erro ao salvar o gráfico '{nome}': {e}")
        caminho = None
    finally:
        plt.close()
    return caminho

def plot_polos_zeros(FT):
    fig, ax = plt.subplots()
    poles = FT.poles()
    zeros = FT.zeros()

    if poles.size > 0:
        ax.scatter(np.real(poles), np.imag(poles), marker='x', color='red', s=100, label='Polos')
    if zeros.size > 0:
        ax.scatter(np.real(zeros), np.imag(zeros), marker='o', color='blue', s=100, facecolors='none', edgecolors='blue', label='Zeros')
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_title('Diagrama de Polos e Zeros')
    ax.legend()
    ax.grid(True)
    
    all_coords_real = np.concatenate((np.real(poles), np.real(zeros)))
    all_coords_imag = np.concatenate((np.imag(poles), np.imag(zeros)))

    if all_coords_real.size > 0 and all_coords_imag.size > 0:
        min_re, max_re = all_coords_real.min(), all_coords_real.max()
        min_im, max_im = all_coords_imag.min(), all_coords_imag.max()

        margin_re = max(0.5, (max_re - min_re) * 0.1)
        margin_im = max(0.5, (max_im - min_im) * 0.1)

        ax.set_xlim(min_re - margin_re, max_re + margin_re)
        ax.set_ylim(min_im - margin_im, max_im + margin_im)
    else:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

    ax.set_aspect('equal', adjustable='box')
    caminho = os.path.join('static', 'polos_zeros.png')
    try:
        os.makedirs('static', exist_ok=True) 
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
    is_admin = session.get('is_admin', False)
    return render_template('index.html', user_email=user_email, admin=is_admin)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        senha = request.form['senha']

        # AVISO DE SEGURANÇA CRÍTICO: Credenciais de admin hardcoded.
        # Em produção, o admin deve ser gerenciado de forma segura (ex: variáveis de ambiente, banco de dados).
        if email == 'tisaaceng@gmail.com' and senha == '4839AT81':
            session['usuario_logado'] = email
            session['is_admin'] = True
            flash('Login de administrador realizado com sucesso!', 'success')
            return redirect(url_for('admin'))

        usuarios = {}
        if os.path.exists('usuarios.json'):
            try:
                with open('usuarios.json', 'r') as f:
                    usuarios = json.load(f)
            except json.JSONDecodeError:
                flash("Erro ao carregar dados de usuários. Arquivo 'usuarios.json' pode estar corrompido.", 'danger')
                usuarios = {} 

        if email in usuarios:
            # AVISO DE SEGURANÇA CRÍTICO: Senhas armazenadas em texto plano.
            # Use hashing de senhas (ex: Flask-Bcrypt ou werkzeug.security.generate_password_hash/check_password_hash)
            # Nunca armazene senhas em texto plano em produção.
            if usuarios[email]['senha'] == senha:
                if usuarios[email].get('aprovado', False):
                    session['usuario_logado'] = email
                    session['is_admin'] = False
                    flash('Login realizado com sucesso!', 'success')
                    return redirect(url_for('painel'))
                else:
                    flash('Seu cadastro ainda não foi aprovado. Por favor, aguarde a aprovação do administrador.', 'warning')
                    return redirect(url_for('login'))
            else:
                flash('Credenciais inválidas. Verifique seu email e senha.', 'danger')
                return redirect(url_for('login'))
        else:
            flash('Credenciais inválidas. Verifique seu email e senha.', 'danger')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/cadastro', methods=['GET', 'POST'])
def cadastro():
    if request.method == 'POST':
        nome = request.form['nome']
        email = request.form['email']
        senha = request.form['senha']

        usuarios = {}
        if os.path.exists('usuarios.json'):
            try:
                with open('usuarios.json', 'r') as f:
                    usuarios = json.load(f)
            except json.JSONDecodeError:
                flash("Erro ao carregar dados de usuários. Arquivo 'usuarios.json' pode estar corrompido. Tente novamente.", 'danger')
                return redirect(url_for('cadastro'))

        if email in usuarios or email == 'tisaaceng@gmail.com':
            flash('Este email já está cadastrado ou reservado.', 'warning')
            return redirect(url_for('cadastro'))

        # AVISO DE SEGURANÇA CRÍTICO: Senhas armazenadas em texto plano.
        # Use hashing de senhas antes de salvar.
        usuarios[email] = {'nome': nome, 'senha': senha, 'aprovado': False}
        
        try:
            with open('usuarios.json', 'w') as f:
                json.dump(usuarios, f, indent=4)
            flash('Cadastro enviado para aprovação. Você será notificado após a aprovação.', 'info')
            return redirect(url_for('login'))
        except IOError as e:
            flash(f"Erro ao salvar o cadastro: {e}. Tente novamente mais tarde.", 'danger')
            return redirect(url_for('cadastro'))
            
    return render_template('cadastro.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if 'usuario_logado' not in session or not session.get('is_admin'):
        flash('Acesso negado. Apenas o administrador pode acessar esta página.', 'danger')
        return redirect(url_for('login'))

    usuarios = {}
    if os.path.exists('usuarios.json'):
        try:
            with open('usuarios.json', 'r') as f:
                usuarios = json.load(f)
        except json.JSONDecodeError:
            flash("Erro ao carregar dados de usuários. Arquivo 'usuarios.json' pode estar corrompido.", 'danger')
            usuarios = {}

    if request.method == 'POST':
        email_to_approve = request.form.get('email')
        if email_to_approve and email_to_approve in usuarios:
            usuarios[email_to_approve]['aprovado'] = True
            try:
                with open('usuarios.json', 'w') as f:
                    json.dump(usuarios, f, indent=4)
                flash(f'Usuário {email_to_approve} aprovado com sucesso!', 'success')
            except IOError as e:
                flash(f"Erro ao salvar aprovação: {e}.", 'danger')
        else:
            flash('Usuário não encontrado para aprovação.', 'danger')

    nao_aprovados = {k: v for k, v in usuarios.items() if not v.get('aprovado', False)}
    return render_template('admin.html', usuarios=nao_aprovados, admin=True)

@app.route('/painel')
def painel():
    if 'usuario_logado' not in session:
        flash('Acesso negado. Por favor, faça login.', 'danger')
        return redirect(url_for('login'))
    is_admin = session.get('is_admin', False)
    return render_template('painel.html', admin=is_admin)

@app.route('/logout')
def logout():
    session.pop('usuario_logado', None)
    session.pop('is_admin', None)
    flash('Você saiu da sua conta.', 'info')
    return redirect(url_for('login'))

@app.route('/simulador', methods=['GET', 'POST'])
def simulador():
    if 'usuario_logado' not in session:
        flash('Faça login para acessar o simulador.', 'warning')
        return redirect(url_for('login'))

    email = session['usuario_logado']

    if not session.get('is_admin', False):
        usuarios = {}
        if os.path.exists('usuarios.json'):
            try:
                with open('usuarios.json', 'r') as f:
                    usuarios = json.load(f)
            except json.JSONDecodeError:
                flash("Erro ao carregar dados de usuários para verificação. Arquivo 'usuarios.json' pode estar corrompido.", 'danger')
                return redirect(url_for('painel')) 

        if not usuarios.get(email, {}).get('aprovado', False):
            flash('Seu cadastro ainda não foi aprovado para usar o simulador. Por favor, aguarde a aprovação do administrador.', 'warning')
            return redirect(url_for('painel'))
    
    resultado = error = None
    is_admin = session.get('is_admin', False)

    if request.method == 'POST':
        edo = request.form.get('edo')
        entrada = request.form.get('entrada')
        saida = request.form.get('saida')

        if not edo or not entrada or not saida:
            error = "Por favor, preencha todos os campos: Equação Diferencial, Variável de Entrada e Variável de Saída."
            flash(error, 'danger')
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

                def tf_to_sympy_tf(tf_control_obj):
                    num_list = tf_control_obj.num[0][0]
                    den_list = tf_control_obj.den[0][0]
                    s_sym = sp.symbols('s')
                    num_poly = sum(coef * s_sym**(len(num_list)-i-1) for i, coef in enumerate(num_list))
                    den_poly = sum(coef * s_sym**(len(den_list)-i-1) for i, coef in enumerate(den_list))
                    return num_poly / den_poly

                expr_pid = sp.simplify(tf_to_sympy_tf(pid))
                expr_mf = sp.simplify(tf_to_sympy_tf(mf))

                img_resposta_aberta = salvar_grafico_resposta(t_open, y_open, 'resposta_malha_aberta', rotacao=0)
                img_resposta_fechada = salvar_grafico_resposta(t_closed, y_closed, 'resposta_malha_fechada', rotacao=0, deslocamento=0.0)
                img_pz = plot_polos_zeros(FT)

                den_coefs = flatten_and_convert(FT.den[0])
                routh_table = tabela_routh(den_coefs)

                resultado = {
                    'ft_latex': ft_latex,
                    'pid_latex': sp.latex(expr_pid, mul_symbol='dot'),
                    'mf_latex': sp.latex(expr_mf, mul_symbol='dot'),
                    'Kp': Kp,
                    'Ki': Ki,
                    'Kd': Kd,
                    'img_resposta_aberta': img_resposta_aberta if img_resposta_aberta else '',
                    'img_resposta_fechada': img_resposta_fechada if img_resposta_fechada else '',
                    'img_pz': img_pz if img_pz else '',
                    'routh_table': routh_table.tolist()
                }

            except ValueError as ve:
                error = f"Erro de validação ou cálculo: {str(ve)}"
                flash(error, 'danger')
            except Exception as e:
                error = f"Um erro inesperado ocorreu durante o processamento: {str(e)}. Por favor, tente novamente ou verifique a EDO."
                flash(error, 'danger')
    
    return render_template('simulador.html', resultado=resultado, error=error, admin=is_admin)

@app.route('/perfil')
def perfil():
    if 'usuario_logado' not in session:
        flash('Faça login para acessar o perfil.', 'warning')
        return redirect(url_for('login'))

    email = session['usuario_logado']
    usuarios = {}
    if os.path.exists('usuarios.json'):
        try:
            with open('usuarios.json', 'r') as f:
                usuarios = json.load(f)
        except json.JSONDecodeError:
            flash("Erro ao carregar dados de usuários. Arquivo 'usuarios.json' pode estar corrompido.", 'danger')
            usuarios = {}

    usuario = usuarios.get(email)
    is_admin = session.get('is_admin', False)
    return render_template('perfil.html', usuario=usuario, email=email, admin=is_admin)

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
        usuarios = {}
        if os.path.exists('usuarios.json'):
            try:
                with open('usuarios.json', 'r') as f:
                    usuarios = json.load(f)
            except json.JSONDecodeError:
                flash("Erro ao carregar dados de usuários. Arquivo 'usuarios.json' pode estar corrompido.", 'danger')
                usuarios = {}

        if email not in usuarios:
            flash('Usuário não encontrado. Por favor, tente novamente.', 'danger')
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
        try:
            with open('usuarios.json', 'w') as f:
                json.dump(usuarios, f, indent=4)
            flash('Senha alterada com sucesso!', 'success')
            return redirect(url_for('perfil'))
        except IOError as e:
            flash(f"Erro ao salvar a nova senha: {e}.", 'danger')
            return redirect(url_for('alterar_senha'))

    is_admin = session.get('is_admin', False)
    return render_template('alterar_senha.html', admin=is_admin)

@app.route('/funcao_transferencia')
def funcao_transferencia():
    flash("Esta página atualmente não exibe a última Função de Transferência calculada. Use o simulador.", 'info')
    ft_latex = "Função de Transferência não disponível ou não calculada recentemente."
    is_admin = session.get('is_admin', False)
    return render_template('transferencia.html', ft_latex=ft_latex, admin=is_admin)

# === EXECUÇÃO PRINCIPAL ===
if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)
