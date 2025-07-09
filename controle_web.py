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
app.secret_key = 'sua_chave_secreta' # Mantenha esta chave secreta e única em produção

# === FUNÇÕES AUXILIARES ===

def flatten_and_convert(lst):
    """
    Achata uma lista e tenta converter seus elementos para float.
    Levanta um erro se um elemento simbólico não puder ser avaliado.
    """
    result = []
    for c in lst:
        if isinstance(c, (list, tuple, np.ndarray)):
            result.extend(flatten_and_convert(c))
        elif hasattr(c, 'iter') and not isinstance(c, (str, bytes)): # Evita iterar sobre strings
            result.extend(flatten_and_convert(c))
        else:
            try:
                result.append(float(c))
            except TypeError:
                # Captura erro se o símbolo não puder ser avaliado para float
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
    s = sp.symbols('s') # Garante que 's' está no escopo para este parse

    # Define os símbolos Laplace (Xs, Fs)
    Xs = sp.symbols(f'{saida_str}s')
    Fs = sp.symbols(f'{entrada_str}s')

    # Cria um dicionário local para o sp.sympify para interpretar a EDO
    _local_sympify_map = {
        'sp': sp, 't': t,
        saida_str: sp.Function(saida_str)(t),
        entrada_str: sp.Function(entrada_str)(t),
    }
    for i in range(1, 5): # Mapeia derivadas para sp.Derivative no parsing
        _local_sympify_map[f'diff({saida_str},t,{i})'] = sp.Derivative(sp.Function(saida_str)(t), t, i)
        _local_sympify_map[f'diff({entrada_str},t,{i})'] = sp.Derivative(sp.Function(entrada_str)(t), t, i)
    _local_sympify_map[f'diff({saida_str},t)'] = sp.Derivative(sp.Function(saida_str)(t), t, 1)
    _local_sympify_map[f'diff({entrada_str},t)'] = sp.Derivative(sp.Function(entrada_str)(t), t, 1)

    # Processa a string da EDO (igualando a zero)
    eq_str_processed = edo_str.replace('diff', 'sp.Derivative')
    if '=' in eq_str_processed:
        lhs, rhs = eq_str_processed.split('=')
        eq_str_processed = f"({lhs.strip()}) - ({rhs.strip()})"
    
    try:
        # Sympify a EDO. `eq_sym` conterá os objetos SymPy `Function` e `Derivative`
        eq_sym = sp.sympify(eq_str_processed, locals=_local_sympify_map)
    except Exception as e:
        raise ValueError(f"Erro ao interpretar a EDO. Verifique a sintaxe. Detalhes: {e}")

    # --- NOVA ABORDAGEM: RECONSTRUÇÃO DA EQUAÇÃO NO DOMÍNIO DE LAPLACE ---
    expr_laplace = sp.S.Zero # Inicializa a expressão de Laplace como zero simbólico
    
    try:
        # Itera sobre os termos da equação simbólica (ex: a*diff(x,t,2), b*diff(x,t), c*x, d*F)
        # sp.Add.make_args(eq_sym) decompõe a equação em termos somados
        for term in sp.Add.make_args(eq_sym):
            # Tenta separar o coeficiente do termo de função/derivada.
            # Se o termo é apenas uma função ou derivada (sem coeficiente explícito), coeff_part será 1 e term_part será a função/derivada
            # Se o termo é uma constante ou parâmetro (sem função/derivada), term.as_coeff_Mul() pode retornar o próprio termo como coeff_part e 1 como term_part
            coeff_part, term_part = term.as_coeff_Mul() 

            if isinstance(term_part, (sp.Function, sp.Derivative)):
                # Se o termo é uma função ou derivada do tempo
                if isinstance(term_part, sp.Function) and term_part.args == (t,):
                    # Termos como x(t) ou F(t)
                    if str(term_part.func) == saida_str:
                        expr_laplace += coeff_part * Xs
                    elif str(term_part.func) == entrada_str:
                        expr_laplace += coeff_part * Fs
                    else:
                        raise ValueError(f"Termo no domínio do tempo não reconhecido como entrada/saída: {term}. A EDO deve conter apenas as variáveis de entrada/saída definidas.")

                elif isinstance(term_part, sp.Derivative):
                    # Termos como diff(x,t) ou diff(F,t,2)
                    base_func = term_part.expr # A função sendo derivada (e.g., x(t))
                    order = term_part.derivative_count # Ordem da derivada (e.g., 1, 2)
                    
                    if isinstance(base_func, sp.Function) and base_func.args == (t,):
                        if str(base_func.func) == saida_str:
                            expr_laplace += coeff_part * (s**order * Xs)
                        elif str(base_func.func) == entrada_str:
                            expr_laplace += coeff_part * (s**order * Fs)
                        else:
                            raise ValueError(f"Derivada de termo não reconhecido como entrada/saída: {term}. A EDO deve conter apenas as variáveis de entrada/saída definidas.")
                    else:
                        raise ValueError(f"Formato de derivada inválido no termo: {term}. Funções devem ser do tipo f(t).")
            else:
                # Se não é uma função/derivada, é um termo constante ou um parâmetro simbólico.
                # Deve ser adicionado à expressão de Laplace como está.
                expr_laplace += term

    except Exception as e:
        raise ValueError(f"Erro na reconstrução da EDO para o domínio de Laplace. Detalhes: {e}. Isso pode indicar um problema na decomposição dos termos da EDO.")

    # --- Verificação e Isolamento da FT (o mesmo que nas últimas versões) ---
    try:
        # Coeficiente do termo de saída Xs (será o denominador da FT)
        den_poly_sym = expr_laplace.coeff(Xs) 
        
        # Coeficiente do termo de entrada Fs (será o numerador da FT, depois de mover para o outro lado)
        num_poly_sym_raw = expr_laplace.coeff(Fs) 
        
        # Verifica se há termos constantes ou não lineares remanescentes
        constant_term = expr_laplace - den_poly_sym * Xs - num_poly_sym_raw * Fs
        
        if constant_term != 0:
            raise ValueError(f"A equação transformada para Laplace contém termos constantes ou não lineares: {constant_term}. A Função de Transferência é aplicável apenas a EDOs lineares com condições iniciais zero.")

        if den_poly_sym == 0:
             raise ValueError("O coeficiente da variável de saída (Xs) no denominador é zero, indicando uma Função de Transferência inválida ou EDO mal formada (ex: a saída não depende de si mesma ou a EDO é apenas uma constante).")
        
        if num_poly_sym_raw == 0 and Fs not in expr_laplace.free_symbols: # Se não há Fs na expressão transformada
            raise ValueError("Não foi possível identificar a variável de entrada (Fs) na equação transformada para Laplace. Verifique a EDO e a variável de entrada. (Ex: F não está presente na EDO ou tem coeficiente zero).")

        # A Função de Transferência G(s) = Xs/Fs.
        # Se a equação é (den_poly_sym)*Xs + (num_poly_sym_raw)*Fs = 0,
        # então (den_poly_sym)*Xs = -(num_poly_sym_raw)*Fs
        # Xs/Fs = -(num_poly_sym_raw) / (den_poly_sym)
        Ls_expr = sp.simplify(-num_poly_sym_raw / den_poly_sym)

    except Exception as e:
        raise ValueError(f"Não foi possível isolar a Função de Transferência. Verifique a EDO e as variáveis de entrada/saída. Erro: {e}")

    num, den = sp.fraction(Ls_expr)
    
    # Identifica quaisquer símbolos remanescentes (coeficientes simbólicos)
    simbolos_num = list(num.free_symbols - {s})
    simbolos_den = list(den.free_symbols - {s})
    
    # Se houver símbolos, avalia-os para 1.0 para permitir cálculos numéricos e gráficos
    if simbolos_num or simbolos_den:
        subs_dict = {sym: 1.0 for sym in simbolos_num + simbolos_den}
        num_eval = num.subs(subs_dict)
        den_eval = den.subs(subs_dict)
        flash(f"Aviso: Coeficientes simbólicos '{', '.join(map(str, simbolos_num + simbolos_den))}' foram detectados e substituídos por 1.0 para a geração dos gráficos e parâmetros PID. Para resultados exatos, forneça apenas coeficientes numéricos na EDO ou revise a sintaxe.", 'warning')
    else:
        num_eval = num
        den_eval = den

    # Converte os coeficientes simbólicos (agora avaliados) para floats
    try:
        # Garante que os objetos são SymPy Poly para extrair coeficientes
        poly_num = sp.Poly(num_eval, s)
        poly_den = sp.Poly(den_eval, s)
        
        num_coeffs = [float(c.evalf()) for c in poly_num.all_coeffs()]
        den_coeffs = [float(c.evalf()) for c in poly_den.all_coeffs()]
    except Exception as e:
        raise ValueError(f"Erro ao extrair coeficientes numéricos da FT: {e}. Certifique-se de que a FT resultante seja válida para conversão.")

    # Remove zeros iniciais dos denominadores (se Poly.all_coeffs() gerar [0, a, b])
    # Isso é importante antes de criar a control.TransferFunction
    while len(den_coeffs) > 1 and abs(den_coeffs[0]) < 1e-9:
        den_coeffs.pop(0)
        # Ajusta o numerador junto se a ordem também reduzir, ou preenche com zero
        if num_coeffs and len(num_coeffs) > 0 and abs(num_coeffs[0]) < 1e-9:
            num_coeffs.pop(0)
        else: # Se o numerador é menor que o denominador após a remoção de zeros, preenche com zeros à esquerda
            num_coeffs = [0] * (len(den_coeffs) - len(num_coeffs)) + num_coeffs
    
    # Pad e verifica zero de novo, após a conversão para numéricos
    num_coeffs, den_coeffs = pad_coeffs(num_coeffs, den_coeffs)
    
    if not den_coeffs or all(c == 0 for c in den_coeffs):
        raise ValueError("O denominador da função de transferência resultou em zero ou está vazio após a avaliação numérica. Verifique a equação diferencial.")
    
    # Se o primeiro coeficiente do denominador ainda for zero, é um problema.
    if abs(den_coeffs[0]) < 1e-9 and len(den_coeffs) > 1:
        # Tenta uma última limpeza se ainda houver zero líder após o padding
        while len(den_coeffs) > 1 and abs(den_coeffs[0]) < 1e-9:
            den_coeffs.pop(0)
            if num_coeffs and len(num_coeffs) > 0: # Ajusta o numerador se necessário
                num_coeffs.pop(0)
        
        if not den_coeffs or abs(den_coeffs[0]) < 1e-9: # Se ainda assim o den for inválido
            raise ValueError("O denominador da função de transferência é inválido ou resultou em zero após simplificação e limpeza.")
    
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
        # Aumentar o tempo de simulação para sistemas instáveis ou lentos
        if FT.poles() is not None and np.any(np.real(FT.poles()) >= 0): # Se tiver polos no semi-plano direito ou no eixo imaginário
            if tempo[-1] < 20: # Aumenta o tempo se a FT for instável e o tempo atual for curto
                tempo = np.linspace(0, 20, 2000) # Simula por mais tempo
        
        t, y = step(FT, T=tempo)
        
        # Verifica se a resposta explodiu (valores muito grandes)
        if np.any(np.abs(y) > 1e10): # Limite arbitrário para "explodiu"
            flash("Aviso: A resposta ao degrau apresentou valores muito grandes, indicando um sistema instável. O gráfico pode ser difícil de interpretar.", 'warning')
            # Pode-se opcionalmente truncar os valores para evitar que o gráfico fique inútil
            y[np.abs(y) > 1e10] = np.sign(y[np.abs(y) > 1e10]) * 1e10

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

    # Se a resposta é essencialmente constante ou não varia, L e T são pequenos.
    if abs(y_final - y_inicial) < 1e-6:
        return 0.01, 0.01

    # Normaliza a resposta entre 0 e 1 (ou -1 e 0 se for decrescente)
    y_scaled = (y - y_inicial) / (y_final - y_inicial)

    try:
        # Encontra o primeiro ponto onde a resposta começa a subir (acima de um pequeno limiar)
        indice_inicio = next(i for i, v in enumerate(y_scaled) if v > 0.01 or v < -0.01)
        L = t[indice_inicio]
    except StopIteration:
        L = 0.01 # Pequeno atraso se não houver mudança significativa

    y_63_target = 0.63
    try:
        # T - Tempo em que a resposta atinge 63% da variação total, subtraído de L
        indice_63 = next(i for i, v in enumerate(y_scaled) if v >= y_63_target)
        T = t[indice_63] - L
    except StopIteration:
        T = 0.01 # Pequena constante de tempo se não atingir 63%

    return (L if L >= 0 else 0.01), (T if T >= 0 else 0.01)

def sintonia_ziegler_nichols(L, T):
    """Calcula os parâmetros PID (Kp, Ki, Kd) usando as regras de Ziegler-Nichols."""
    if L == 0: L = 1e-6 # Evita divisão por zero
    if T == 0: T = 1e-6 # Evita divisão por zero
    
    Kp = 1.2 * T / L
    Ti = 2 * L
    Td = 0.5 * L
    
    Ki = Kp / Ti if Ti != 0 else 0.0
    Kd = Kp * Td
    return Kp, Ki, Kd

def cria_pid_tf(Kp, Ki, Kd):
    """Cria a Função de Transferência de um controlador PID."""
    s_tf = control.TransferFunction.s
    return Kp + Ki / s_tf + Kd * s_tf

def malha_fechada_tf(Gp, Gc):
    """Calcula a Função de Transferência de malha fechada."""
    return control.feedback(Gp * Gc, 1)

def tabela_routh(coeficientes):
    """
    Gera a tabela de Routh-Hurwitz para análise de estabilidade.
    """
    try:
        coeficientes = [float(c) for c in coeficientes]
    except TypeError as e:
        raise ValueError(f"Erro ao processar coeficientes para a Tabela de Routh. Verifique se todos os coeficientes são numéricos. Detalhes: {e}. Coeficientes recebidos: {coeficientes}")
    except Exception as e:
        raise ValueError(f"Erro inesperado ao processar coeficientes para a Tabela de Routh. Detalhes: {e}. Coeficientes recebidos: {coeficientes}")

    n = len(coeficientes)
    
    if n == 0 or all(c == 0 for c in coeficientes):
        flash("Aviso: Coeficientes do denominador são todos zero ou vazios para a Tabela de Routh.", 'warning')
        return np.array([[]])

    while n > 0 and abs(coeficientes[0]) < 1e-9: # Remove zeros iniciais
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
        if abs(routh[i - 1, 0]) < 1e-9: # Lida com zeros na primeira coluna
            routh[i - 1, 0] = 1e-9
            flash(f"Aviso: Zero ou valor muito próximo de zero detectado na primeira coluna da linha {i-1} da Tabela de Routh. Substituindo por um pequeno valor (1e-9) para continuar o cálculo. Isso pode indicar polos no eixo imaginário ou casos especiais de estabilidade.", 'warning')

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
    """
    Salva um gráfico de resposta ao degrau.
    Permite rotação e deslocamento dos dados.
    """
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
        t, y = y, t
        t = -t
        y = y - y.min()
    
    # Garante que os valores de tempo e saída não são negativos para plotagem
    if t.size > 0 and min(t) < 0:
        t = t - min(t)
    if y.size > 0 and min(y) < 0:
        y = y - min(y)

    plt.figure(figsize=(8, 4))
    plt.plot(t, y, label='Resposta ao Degrau')
    plt.xlabel('Tempo (s)' if rotacao not in [90, 270] else 'Saída')
    plt.ylabel('Saída' if rotacao not in [90, 270] else 'Tempo (s)')
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
    """Cria e salva o diagrama de polos e zeros."""
    try:
        poles = FT.poles()
        zeros = FT.zeros()
    except Exception as e:
        print(f"Erro ao obter polos e zeros da FT: {e}")
        flash("Não foi possível determinar os polos e zeros da Função de Transferência.", 'warning')
        return None

    fig, ax = plt.subplots()
    
    if poles.size > 0:
        ax.scatter(np.real(poles), np.imag(poles), marker='x', color='red', label='Polos', s=100)
    if zeros.size > 0:
        ax.scatter(np.real(zeros), np.imag(zeros), marker='o', color='blue', label='Zeros', s=100)
        
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Parte Real')
    ax.set_ylabel('Parte Imaginária')
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
    is_admin = session.get('is_admin', False) 
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

        # Admin fixo
        if email == 'tisaaceng@gmail.com' and senha == '4839AT81':
            session['usuario_logado'] = email
            session['is_admin'] = True
            flash('Login de administrador realizado com sucesso!', 'success')
            return redirect(url_for('admin')) # Admin vai para o painel admin

        # Usuários comuns
        if email in usuarios and usuarios[email]['senha'] == senha:
            if usuarios[email].get('aprovado', False):
                session['usuario_logado'] = email
                session['is_admin'] = False
                flash('Login realizado com sucesso!', 'success')
                return redirect(url_for('painel')) # Usuário comum vai para o painel de usuário
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

    if os.path.exists('usuarios.json'):
        with open('usuarios.json', 'r') as f:
            usuarios = json.load(f)
        nao_aprovados = {k: v for k, v in usuarios.items() if not v.get('aprovado', False)}
    else:
        nao_aprovados = {}

    return render_template('admin.html', usuarios=nao_aprovados, admin=True)

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

    # Se não for admin, verifica aprovação
    if not session.get('is_admin', False): # Use session.get('is_admin', False) para verificar
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
                
                t_open, y_open = resposta_degrau(FT)
                L, T = estima_LT(t_open, y_open)
                Kp, Ki, Kd = sintonia_ziegler_nichols(L, T)
                pid = cria_pid_tf(Kp, Ki, Kd)
                mf = malha_fechada_tf(FT, pid)
                t_closed, y_closed = resposta_degrau(mf)

                def tf_to_sympy_tf(tf_control_obj):
                    num_list = tf_control_obj.num[0][0]
                    den_list = tf_control_obj.den[0][0]

                    # Cria polinômios SymPy a partir dos coeficientes
                    num_poly = sum(coef * s**(len(num_list) - i - 1) for i, coef in enumerate(num_list))
                    den_poly = sum(coef * s**(len(den_list) - i - 1) for i, coef in enumerate(den_list))
                    return num_poly / den_poly

                expr_pid = sp.simplify(tf_to_sympy_tf(pid))
                expr_mf = sp.simplify(tf_to_sympy_tf(mf))

                img_resposta_aberta = salvar_grafico_resposta(t_open, y_open, 'resposta_malha_aberta', rotacao=0)
                img_resposta_fechada = salvar_grafico_resposta(t_closed, y_closed, 'resposta_malha_fechada', rotacao=0)
                img_pz = plot_polos_zeros(FT)

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

    is_admin = session.get('is_admin', False)
    return render_template('alterar_senha.html', admin=is_admin)

@app.route('/funcao_transferencia')
def funcao_transferencia():
    ft_latex = session.get('ft_latex')
    if not ft_latex:
        ft_latex = "Função de Transferência não disponível ou não calculada recentemente."
    is_admin = session.get('is_admin', False)
    return render_template('transferencia.html', ft_latex=ft_latex, admin=is_admin)


# === EXECUÇÃO PRINCIPAL ===

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)
