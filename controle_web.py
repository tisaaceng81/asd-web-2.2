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
app.secret_key = 'sua_chave_secreta' # Mantenha esta chave secreta e única em produção

# === FUNÇÕES AUXILIARES ===

def flatten_and_convert(lst):
    """
    Achata uma lista e tenta converter seus elementos para float.
    Levanta um erro se um elemento simbólico não puder ser avaliado.
    """
    result = []
    for c in lst:
        # Usar isinstance para Numpy arrays e outras iteráveis para maior robustez
        if isinstance(c, (list, tuple, np.ndarray)) or (hasattr(c, '__iter__') and not isinstance(c, (str, bytes))):
            result.extend(flatten_and_convert(c))
        else:
            try:
                result.append(float(c))
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
    t = sp.symbols('t', real=True)
    # Criar funções simbólicas de referência (não serão usadas diretamente para comparação por identidade)
    x_func_ref = sp.Function(saida_str)(t)
    F_func_ref = sp.Function(entrada_str)(t)

    eq_str = edo_str.replace('diff', 'sp.Derivative')
    if '=' in eq_str:
        lhs, rhs = eq_str.split('=')
        eq_str = f"({lhs.strip()}) - ({rhs.strip()})"

    # local_dict: Mapeia o nome da string da variável para o objeto Function
    # Isso ajuda o SymPy a parsear a EDO corretamente.
    local_dict = {
        'sp': sp, 't': t,
        saida_str: x_func_ref, 
        entrada_str: F_func_ref,
        # Mantido do seu código original para compatibilidade com strings literais
        'F': F_func_ref, 'x': x_func_ref,
        str(F_func_ref): F_func_ref, str(x_func_ref): x_func_ref 
    }
    
    # Adicionar derivadas ao local_dict para sympify reconhecer
    for i in range(1, 5): # Suporta até a 4ª derivada
        local_dict[f'diff({saida_str},t,{i})'] = sp.Derivative(x_func_ref, t, i)
        local_dict[f'diff({entrada_str},t,{i})'] = sp.Derivative(F_func_ref, t, i)
    local_dict[f'diff({saida_str},t)'] = sp.Derivative(x_func_ref, t, 1)
    local_dict[f'diff({entrada_str},t)'] = sp.Derivative(F_func_ref, t, 1)


    eq = sp.sympify(eq_str, locals=local_dict)

    # Definir símbolos Laplace dinamicamente
    Xs = sp.symbols(f'{saida_str}s')
    Fs = sp.symbols(f'{entrada_str}s')

    expr_laplace = eq
    
    # Lógica de substituição de Laplace: Iterar sobre os átomos da expressão original
    # e substituir com base nos NOMES das funções e suas ordens de derivação.
    # Isso é mais robusto contra problemas de identidade de objeto do SymPy.
    
    subs_map = {}
    for atom in expr_laplace.atoms():
        # Lida com funções base (e.g., x(t), F(t))
        if isinstance(atom, sp.Function) and atom.args == (t,):
            if str(atom.func) == saida_str: # Comparar por nome da função
                subs_map[atom] = Xs
            elif str(atom.func) == entrada_str: # Comparar por nome da função
                subs_map[atom] = Fs
        # Lida com derivadas (e.g., diff(x,t,2))
        elif isinstance(atom, sp.Derivative) and atom.expr.args == (t,):
            base_func_of_deriv = atom.expr 
            order = atom.derivative_count
            
            if str(base_func_of_deriv.func) == saida_str: # Comparar por nome da função base
                subs_map[atom] = s**order * Xs
            elif str(base_func_of_deriv.func) == entrada_str: # Comparar por nome da função base
                subs_map[atom] = s**order * Fs
    
    expr_laplace = expr_laplace.subs(subs_map)

    # Validação pós-substituição (melhoria mantida e aprimorada)
    for atom in expr_laplace.atoms():
        if (isinstance(atom, sp.Function) and atom.args == (t,)) or \
           (isinstance(atom, sp.Derivative) and atom.expr.args == (t,)):
            raise ValueError(f"Erro na transformação de Laplace: A equação ainda contém termos no domínio do tempo como '{atom}'. Verifique a EDO e as variáveis de entrada/saída fornecidas. Isso pode indicar uma incompatibilidade na sintaxe ou nos nomes das variáveis.")


    # Lógica de isolamento da FT (do seu código original funcional, com validações adicionais)
    try:
        lhs = expr_laplace
        coef_Xs = lhs.coeff(Xs) # AQUI é onde falhava antes, se Xs não era identificado corretamente

        # Validar se o coeficiente de Xs é zero ANTES de calcular o resto
        if coef_Xs == 0:
            raise ValueError("O coeficiente da variável de saída (Xs) no denominador é zero, indicando uma Função de Transferência inválida ou EDO mal formada (ex: a saída não depende de si mesma ou a EDO é apenas uma constante).")
        
        resto = lhs - coef_Xs * Xs

        # Validar se o termo restante contém Fs, se não, EDO pode estar incompleta
        if Fs not in resto.free_symbols: # Se Fs não está no resto
            if resto != 0: # E o resto não é zero, significa termos não-Fs e não-Xs
                 raise ValueError(f"Não foi possível isolar a Função de Transferência. Termos remanescentes inesperados: {resto}. Verifique se a EDO é linear e homogênea (assumindo CI zero) e se a variável de entrada está presente e correta.")
            # Se resto == 0 e Fs não está lá, significa que não há termo de entrada F(t).
            else:
                 raise ValueError("A EDO não possui uma variável de entrada (Fs). Não é possível formar uma função de transferência X(s)/F(s).")


        Ls_expr = -resto / coef_Xs
        Ls_expr = sp.simplify(Ls_expr.subs(Fs, 1)) # Assumir Fs=1 para a FT

    except Exception as e:
        raise ValueError(f"Não foi possível isolar a Função de Transferência. Verifique a EDO e as variáveis de entrada/saída. Erro: {e}")

    num, den = sp.fraction(Ls_expr)
    
    # Tratamento de coeficientes simbólicos (melhoria mantida da versão anterior)
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
    
    # Padding e limpeza de zeros iniciais (melhorias mantidas)
    num_coeffs, den_coeffs = pad_coeffs(num_coeffs, den_coeffs)
    
    if not den_coeffs or all(c == 0 for c in den_coeffs):
        raise ValueError("O denominador da função de transferência resultou em zero ou está vazio após a avaliação numérica. Verifique a equação diferencial.")
    
    while len(den_coeffs) > 1 and abs(den_coeffs[0]) < 1e-9:
        den_coeffs.pop(0)
        if num_coeffs and len(num_coeffs) > 0 and abs(num_coeffs[0]) < 1e-9:
            num_coeffs.pop(0)
        else: # Se o numerador é menor, preenche com zeros à esquerda
            num_coeffs = [0] * (len(den_coeffs) - len(num_coeffs)) + num_coeffs

    FT = control.TransferFunction(num_coeffs, den_coeffs)
    return Ls_expr, FT

def ft_to_latex(expr):
    return sp.latex(expr, mul_symbol='dot')

def resposta_degrau(FT, tempo=None):
    """
    Calcula a resposta ao degrau de uma Função de Transferência.
    Sua versão original, com melhorias de validação e aviso para sistemas instáveis.
    """
    if tempo is None:
        tempo = np.linspace(0, 10, 1000)
    try: # Início do try block
        # Aumentar o tempo de simulação para sistemas instáveis ou lentos (melhoria mantida)
        if FT.poles() is not None and np.any(np.real(FT.poles()) >= 0):
            if tempo[-1] < 20:
                tempo = np.linspace(0, 20, 2000)
        
        t, y = step(FT, T=tempo) # Linha que pode levantar exceção
        
        # Verifica se a resposta explodiu (melhoria mantida)
        if np.any(np.abs(y) > 1e10):
            flash("Aviso: A resposta ao degrau apresentou valores muito grandes, indicando um sistema instável. O gráfico pode ser difícil de interpretar.", 'warning')
            y[np.abs(y) > 1e10] = np.sign(y[np.abs(y) > 1e10]) * 1e10

        return t, y # Retorno agora dentro do try block
    except Exception as e: # Fim do try block, início do except
        raise ValueError(f"Não foi possível calcular a resposta ao degrau. Pode ser devido a polos instáveis ou erro na função de transferência. Erro: {e}")

def estima_LT(t, y):
    # Sua versão original da estima_LT com pequenas melhorias
    y = np.array(y)
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("A resposta ao degrau contém valores inválidos (NaN/Inf), impossível estimar L e T.")
    
    y_final = y[-1]
    y_inicial = y[0] # Incluído para calcular variação real

    # Se a resposta final é essencialmente zero ou não variou significativamente
    if abs(y_final - y_inicial) < 1e-6:
        return 0.01, 0.01

    # Normalizar a resposta em relação à variação total
    y_scaled = (y - y_inicial) / (y_final - y_inicial)

    try:
        indice_inicio = next(i for i, v in enumerate(y_scaled) if v > 0.01 or v < -0.01)
        L = t[indice_inicio]
    except StopIteration:
        L = 0.01
    
    y_63_target = 0.63
    try:
        indice_63 = next(i for i, v in enumerate(y_scaled) if v >= y_63_target)
        T = t[indice_63] - L
    except StopIteration:
        T = 0.01
    
    return (L if L >= 0 else 0.01), (T if T >= 0 else 0.01) # Garante L e T positivos

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
    # Sua versão original, com melhorias de validação de coeficientes e zeros na primeira coluna
    # A linha de conversão foi feita mais robusta para garantir floats.
    try:
        coeficientes = [float(c) for c in coeficientes]
    except Exception as e:
        raise ValueError(f"Erro ao processar coeficientes para a Tabela de Routh: {e}. Certifique-se de que a entrada é uma lista plana de números.")

    n = len(coeficientes)
    
    if n == 0 or all(c == 0 for c in coeficientes):
        flash("Aviso: Coeficientes do denominador são todos zero ou vazios para a Tabela de Routh.", 'warning')
        return np.array([[]])

    # Remove zeros iniciais se houver (ex: 0s^2 + 2s + 1)
    while n > 0 and abs(coeficientes[0]) < 1e-9:
        coeficientes.pop(0)
        n = len(coeficientes)
    
    if n == 0:
        flash("Aviso: Todos os coeficientes do denominador se anularam após a remoção de zeros iniciais.", 'warning')
        return np.array([[]])

    if n == 1 and abs(coeficientes[0]) < 1e-9:
        flash("Aviso: O único coeficiente do denominador restante é zero.", 'warning')
        return np.array([[]])

    m = (n + 1) // 2
    routh = np.zeros((n, m))
    
    routh[0, :len(coeficientes[::2])] = coeficientes[::2]
    if n > 1:
        routh[1, :len(coeficientes[1::2])] = coeficientes[1::2]
    
    for i in range(2, n):
        # Lida com o caso de zero ou quase zero na primeira coluna substituindo por um pequeno epsilon
        if abs(routh[i - 1, 0]) < 1e-9:
            routh[i - 1, 0] = 1e-9 # Usar um valor muito pequeno
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
    # Sua função original, com validações de dados e tratamento de caminho
    y = np.array(y) # Converte para numpy array para operações
    valid_indices = np.isfinite(t) & np.isfinite(y)
    t = t[valid_indices]
    y = y[valid_indices]

    if len(t) == 0:
        print(f"Aviso: Dados vazios ou inválidos para o gráfico '{nome}'. Não será gerado.")
        return None

    y = y + deslocamento

    # A lógica de rotação para inversão é controlada aqui.
    # Se rotacao=0 for passado (padrão), não haverá rotação/inversão.
    if rotacao == 180:
        t = -t[::-1]
        y = -y[::-1]
    elif rotacao == 90:
        t, y = -y, t

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
    try:
        plt.savefig(caminho)
    except Exception as e:
        print(f"Erro ao salvar o gráfico '{nome}': {e}")
        caminho = None
    finally:
        plt.close()
    return caminho

def plot_polos_zeros(FT):
    # Sua função original, com pequenas melhorias de validação e ajuste de limites
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
    
    # Ajuste de limites para garantir que o gráfico seja sempre visível e não cortado
    all_coords_real = np.concatenate((np.real(FT.poles()), np.real(FT.zeros())))
    all_coords_imag = np.concatenate((np.imag(FT.poles()), np.imag(FT.zeros())))

    if all_coords_real.size > 0 and all_coords_imag.size > 0:
        min_re, max_re = all_coords_real.min(), all_coords_real.max()
        min_im, max_im = all_coords_imag.min(), all_coords_imag.max()

        margin_re = max(0.5, (max_re - min_re) * 0.1)
        margin_im = max(0.5, (max_im - min_im) * 0.1)

        ax.set_xlim(min_re - margin_re, max_re + margin_re)
        ax.set_ylim(min_im - margin_im, max_im + margin_im)
    else: # Mantenho seu 'else' original, mas o problema anterior era da cópia/cola.
        ax.set_xlim(-2, 2) # Limites padrão se não houver polos/zeros finitos
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
    is_admin = session.get('is_admin', False) # Passa admin status
    return render_template('index.html', user_email=user_email, admin=is_admin)

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
            flash('Login de administrador realizado com sucesso!', 'success') # Flash com categoria
            return redirect(url_for('admin'))

        if email in usuarios and usuarios[email]['senha'] == senha:
            if usuarios[email].get('aprovado', False):
                session['usuario_logado'] = email
                session['is_admin'] = False
                flash('Login realizado com sucesso!', 'success') # Flash com categoria
                return redirect(url_for('painel'))
            else:
                flash('Cadastro ainda não aprovado.', 'warning') # Flash com categoria
                return redirect(url_for('login'))
        else:
            flash('Credenciais inválidas.', 'danger') # Flash com categoria
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
            flash('Email já cadastrado ou reservado.', 'warning') # Flash com categoria
            return redirect(url_for('cadastro'))

        usuarios[email] = {'nome': nome, 'senha': senha, 'aprovado': False}
        with open('usuarios.json', 'w') as f:
            json.dump(usuarios, f, indent=4)

        flash('Cadastro enviado para aprovação.', 'info') # Flash com categoria
        return redirect(url_for('login'))
    return render_template('cadastro.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    # Permite acesso somente se estiver logado como admin (email fixo + flag)
    if 'usuario_logado' not in session or session.get('is_admin') != True:
        flash('Acesso negado. Apenas o administrador pode acessar.', 'danger') # Flash com categoria
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
                flash(f'{email} aprovado com sucesso!', 'success') # Flash com categoria
            else:
                flash('Usuário não encontrado.', 'danger') # Flash com categoria
        else:
            flash('Arquivo de usuários não encontrado.', 'warning') # Flash com categoria

    # Mostrar apenas usuários não aprovados
    if os.path.exists('usuarios.json'):
        with open('usuarios.json', 'r') as f:
            usuarios = json.load(f)
        nao_aprovados = {k: v for k, v in usuarios.items() if not v.get('aprovado', False)}
    else:
        nao_aprovados = {}

    return render_template('admin.html', usuarios=nao_aprovados, admin=True) # Passa admin=True

@app.route('/painel')
def painel():
    if 'usuario_logado' not in session:
        flash('Acesso negado.', 'danger') # Flash com categoria
        return redirect(url_for('login'))
    is_admin = session.get('is_admin', False) # Passa admin status
    return render_template('painel.html', admin=is_admin)

@app.route('/logout')
def logout():
    session.pop('usuario_logado', None)
    session.pop('is_admin', None)
    flash('Logout realizado.', 'info') # Flash com categoria
    return redirect(url_for('login'))

@app.route('/simulador', methods=['GET', 'POST'])
def simulador():
    if 'usuario_logado' not in session:
        flash('Faça login para acessar o simulador.', 'warning') # Flash com categoria
        return redirect(url_for('login'))

    email = session['usuario_logado']

    # Verificar se usuário está aprovado
    if not session.get('is_admin', False): # Admin não precisa estar aprovado para acessar o simulador
        if os.path.exists('usuarios.json'):
            with open('usuarios.json', 'r') as f:
                usuarios = json.load(f)
            if not usuarios.get(email, {}).get('aprovado', False):
                flash('Seu cadastro ainda não foi aprovado para usar o simulador.', 'warning') # Flash com categoria
                return redirect(url_for('painel'))
        else:
            flash('Arquivo de usuários não encontrado.', 'warning') # Flash com categoria
            return redirect(url_for('login'))

    resultado = error = None
    is_admin = session.get('is_admin', False) # Passa admin status

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

                def tf_to_sympy_tf(tf_control_obj): # Renomeado para evitar conflito com 'tf' local
                    # Garante que tf_control_obj.num[0][0] e tf_control_obj.den[0][0] são listas
                    num_list = tf_control_obj.num[0][0]
                    den_list = tf_control_obj.den[0][0]

                    s_sym = sp.symbols('s') # Garante 's' simbólico para esta função
                    num_poly = sum(coef * s_sym**(len(num_list)-i-1) for i, coef in enumerate(num_list))
                    den_poly = sum(coef * s_sym**(len(den_list)-i-1) for i, coef in enumerate(den_list))
                    return num_poly / den_poly

                expr_pid = sp.simplify(tf_to_sympy_tf(pid))
                expr_mf = sp.simplify(tf_to_sympy_tf(mf))

                # Salvar gráficos. A rotação agora é 0 para não inverter.
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
                    'img_resposta_aberta': 'resposta_malha_aberta.png', # Caminho fixo
                    'img_resposta_fechada': 'resposta_malha_fechada.png', # Caminho fixo
                    'img_pz': 'polos_zeros.png', # Caminho fixo
                    'routh_table': routh_table.tolist()
                }

            except ValueError as ve:
                error = f"Erro de validação ou cálculo: {str(ve)}"
            except Exception as e:
                error = f"Erro inesperado no processamento: {str(e)}"
    
    return render_template('simulador.html', resultado=resultado, error=error, admin=is_admin)

@app.route('/perfil')
def perfil():
    if 'usuario_logado' not in session:
        flash('Faça login para acessar o perfil.', 'warning') # Flash com categoria
        return redirect(url_for('login'))

    email = session['usuario_logado']
    if os.path.exists('usuarios.json'):
        with open('usuarios.json', 'r') as f:
            usuarios = json.load(f)
        else:
            usuarios = {}

    usuario = usuarios.get(email)
    is_admin = session.get('is_admin', False) # Passa admin status
    return render_template('perfil.html', usuario=usuario, email=email, admin=is_admin)

@app.route('/alterar_senha', methods=['GET', 'POST'])
def alterar_senha():
    if 'usuario_logado' not in session:
        flash('Faça login para alterar a senha.', 'warning') # Flash com categoria
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
            flash('Usuário não encontrado.', 'danger') # Flash com categoria
            return redirect(url_for('alterar_senha'))

        if usuarios[email]['senha'] != senha_atual:
            flash('Senha atual incorreta.', 'danger') # Flash com categoria
            return redirect(url_for('alterar_senha'))

        if nova_senha != confirmar_senha:
            flash('Nova senha e confirmação não conferem.', 'danger') # Flash com categoria
            return redirect(url_for('alterar_senha'))
        
        if not nova_senha:
            flash('A nova senha não pode ser vazia.', 'danger') # Flash com categoria
            return redirect(url_for('alterar_senha'))

        usuarios[email]['senha'] = nova_senha
        with open('usuarios.json', 'w') as f:
            json.dump(usuarios, f, indent=4)

        flash('Senha alterada com sucesso!', 'success') # Flash com categoria
        return redirect(url_for('perfil'))

    is_admin = session.get('is_admin', False) # Passa admin status
    return render_template('alterar_senha.html', admin=is_admin)

@app.route('/funcao_transferencia')
def funcao_transferencia():
    ft_latex = session.get('ft_latex')
    if not ft_latex:
        ft_latex = "Função de Transferência não disponível."
    is_admin = session.get('is_admin', False) # Passa admin status
    return render_template('transferencia.html', ft_latex=ft_latex, admin=is_admin)

# === EXECUÇÃO PRINCIPAL ===
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)
