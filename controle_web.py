from flask import Flask, render_template, request, redirect, url_for, flash, session
import json
import os
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import control
from control.matlab import step
from sympy.abc import s # Import 's' for Laplace variable
import re # Import regex for symbol extraction

app = Flask(__name__)
app.secret_key = 'sua_chave_secreta'

# Garante que o diretório static exista para os gráficos
if not os.path.exists('static'):
    os.makedirs('static')

# === FUNÇÕES AUXILIARES ===

def flatten_and_convert(lst):
    """
    Achata uma lista e tenta converter seus elementos para float.
    Levanta uma exceção se a conversão falhar.
    """
    result = []
    for c in lst:
        if hasattr(c, '__iter__') and not isinstance(c, (str, bytes)):
            result.extend(flatten_and_convert(c))
        else:
            try:
                result.append(float(c))
            except Exception as e:
                # Isso idealmente não deveria ser alcançado se a verificação has_symbolic_coeffs funcionar
                raise Exception(f"Erro convertendo coeficiente para número: {c} ({e}). Verifique se todos os coeficientes são numéricos.")
    return result

def pad_coeffs(num_coeffs, den_coeffs):
    """
    Preenche os coeficientes do numerador e denominador com zeros
    para que tenham o mesmo comprimento.
    """
    len_num = len(num_coeffs)
    len_den = len(den_coeffs)
    if len_num < len_den:
        num_coeffs = [0.0] * (len_den - len_num) + num_coeffs
    elif len_den < len_num:
        den_coeffs = [0.0] * (len_num - len_den) + den_coeffs
    return num_coeffs, den_coeffs

def parse_edo(edo_str, entrada_str, saida_str):
    """
    Analisa uma Equação Diferencial Ordinária (EDO) e a converte em uma
    função de transferência no domínio de Laplace.
    Aceita variáveis de entrada/saída arbitrárias e coeficientes simbólicos.

    Args:
        edo_str (str): A string da EDO (ex: 'diff(y(t),t,2) + 2*diff(y(t),t) + y(t) = diff(u(t),t) + u(t)').
        entrada_str (str): O nome da variável de entrada (ex: 'u').
        saida_str (str): O nome da variável de saída (ex: 'y').

    Returns:
        tuple: (Ls_expr, FT, has_symbolic_coeffs)
            Ls_expr (sympy.Expr): A função de transferência simbólica.
            FT (control.TransferFunction or None): A função de transferência numérica,
                                                  ou None se houver coeficientes simbólicos.
            has_symbolic_coeffs (bool): True se a FT contém coeficientes simbólicos, False caso contrário.
    """
    t = sp.symbols('t', real=True)
    
    # Define as funções de entrada e saída dinamicamente com base nos nomes fornecidos
    X = sp.Function(saida_str)(t)
    F = sp.Function(entrada_str)(t)

    # Prepara a string da equação para sympify, substituindo 'diff' e garantindo formato de equação
    eq_str = edo_str.replace('diff', 'sp.Derivative')
    if '=' in eq_str:
        lhs, rhs = eq_str.split('=')
        eq_str = f"({lhs.strip()}) - ({rhs.strip()})"

    # Identifica todos os potenciais símbolos na string da EDO
    # Usa regex para encontrar sequências de letras que podem ser símbolos (ex: 'a', 'b', 'k')
    # Evita corresponder a palavras-chave como 'diff', 'sp', 'Derivative'
    potential_symbols = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', eq_str))

    # Inicializa o dicionário local para sp.sympify
    local_dict = {
        'sp': sp,
        't': t,
        saida_str: X, # Mapeia o nome da variável de saída do usuário para a função sympy
        entrada_str: F, # Mapeia o nome da variável de entrada do usuário para a função sympy
    }

    # Adiciona outros símbolos identificados ao local_dict como símbolos sympy
    # Exclui palavras-chave e os nomes das variáveis de entrada/saída já tratadas
    excluded_keywords = {'t', 'diff', 'sp', 'Derivative', entrada_str, saida_str}
    for sym_name in potential_symbols:
        if sym_name not in excluded_keywords:
            local_dict[sym_name] = sp.symbols(sym_name)

    # Analisa a EDO usando sympy
    try:
        eq = sp.sympify(eq_str, locals=local_dict)
    except Exception as e:
        raise ValueError(f"Não foi possível analisar a string da EDO. Verifique a sintaxe e as variáveis definidas. Erro: {e}")

    # Transforma a EDO para o domínio de Laplace
    Xs, Fs = sp.symbols(f'{saida_str}s {entrada_str}s') # Símbolos dinâmicos para o domínio de Laplace
    expr_laplace = eq
    
    # Substitui as derivadas pelas suas equivalentes no domínio 's'
    for d in expr_laplace.atoms(sp.Derivative):
        order = d.derivative_count
        func = d.expr
        if func == X:
            expr_laplace = expr_laplace.subs(d, s**order * Xs)
        elif func == F:
            expr_laplace = expr_laplace.subs(d, s**order * Fs)
    
    # Substitui as funções do domínio do tempo pelos símbolos do domínio 's'
    expr_laplace = expr_laplace.subs({X: Xs, F: Fs})

    # Isola Xs/Fs para obter a função de transferência G(s) = Xs / Fs
    try:
        # Coleta os termos com Xs e Fs
        collected_expr = sp.collect(expr_laplace, [Xs, Fs])
        
        # Obtém os coeficientes de Xs e Fs
        coef_Xs = collected_expr.coeff(Xs)
        coef_Fs = collected_expr.coeff(Fs)

        # Garante que o coeficiente de Xs não seja zero para evitar divisão por zero
        if coef_Xs == 0:
            raise ValueError(f"O coeficiente da variável de saída '{saida_str}' no domínio de Laplace é zero. Não é possível formar a função de transferência.")

        # Calcula a função de transferência G(s) = Xs / Fs
        Ls_expr = -coef_Fs / coef_Xs
        Ls_expr = sp.simplify(Ls_expr)
    except Exception as e:
        raise ValueError(f"Não foi possível derivar a função de transferência. Certifique-se de que a EDO relaciona '{saida_str}' e '{entrada_str}'. Erro: {e}")

    num, den = sp.fraction(Ls_expr)

    # NOVO: Verificação mais robusta de coeficientes numéricos
    has_symbolic_coeffs = False
    try:
        num_poly = sp.Poly(num, s)
        den_poly = sp.Poly(den, s)
        
        for coeff in num_poly.all_coeffs() + den_poly.all_coeffs():
            if not coeff.is_number:
                has_symbolic_coeffs = True
                break
    except Exception as poly_e:
        # Se não for um polinômio em 's', provavelmente contém outros símbolos
        has_symbolic_coeffs = True
        # Não levante erro aqui, apenas marque como simbólico para tratamento posterior
        # print(f"Debug: Erro ao formar polinômio, tratando como simbólico: {poly_e}")

    if has_symbolic_coeffs:
        # Se houver coeficientes simbólicos, não podemos criar uma TransferFunction numérica
        return Ls_expr, None, True # Retorna expressão simbólica, None para FT, e flag simbólica
    else:
        # Se não houver coeficientes simbólicos, converte para numérico e cria a TransferFunction
        try:
            num_coeffs = [float(c) for c in num_poly.all_coeffs()]
            den_coeffs = [float(c) for c in den_poly.all_coeffs()]
            num_coeffs, den_coeffs = pad_coeffs(num_coeffs, den_coeffs)
            FT = control.TransferFunction(num_coeffs, den_coeffs)
            return Ls_expr, FT, False # Retorna expressão simbólica, FT, e flag simbólica
        except Exception as e:
            raise ValueError(f"Não foi possível converter os coeficientes da função de transferência para números. Erro: {e}")

def ft_to_latex(expr):
    """Converte uma expressão sympy para a representação LaTeX."""
    return sp.latex(expr, mul_symbol='dot')

def resposta_degrau(FT, tempo=None):
    """Calcula a resposta ao degrau de uma função de transferência."""
    if tempo is None:
        tempo = np.linspace(0, 10, 1000)
    t, y = step(FT, T=tempo)
    return t, y

def estima_LT(t, y):
    """Estima os parâmetros de tempo morto (L) e constante de tempo (T)
    de um sistema de primeira ordem com atraso (FOPTD) a partir da resposta ao degrau.
    """
    if len(y) == 0 or np.isclose(y[-1], 0):
        return 0.01, 0.01 # Retorna valores padrão para evitar divisão por zero ou sistemas sem resposta

    y_final = y[-1]
    # Encontra o índice onde a resposta começa a subir (1% do valor final)
    # Garante que o índice exista e que y_final não seja zero
    if y_final == 0:
        return 0.01, 0.01 # Evita divisão por zero se a resposta final for 0

    try:
        indice_inicio = next(i for i, v in enumerate(y) if v > 0.01 * y_final)
    except StopIteration:
        indice_inicio = 0 # Se não encontrar, assume que começa no tempo 0
    L = t[indice_inicio]

    # Encontra o índice onde a resposta atinge 63% do valor final
    y_63 = 0.63 * y_final
    try:
        indice_63 = next(i for i, v in enumerate(y) if v >= y_63)
    except StopIteration:
        indice_63 = len(y) - 1 # Se não atingir 63%, usa o último ponto

    T = t[indice_63] - L
    return (L if L >= 0 else 0.01), (T if T >= 0 else 0.01) # Garante que L e T sejam não negativos

def sintonia_ziegler_nichols(L, T):
    """Calcula os parâmetros Kp, Ki, Kd para um controlador PID
    usando o método de Ziegler-Nichols (para resposta ao degrau).
    """
    Kp = 1.2 * T / L if L != 0 else 1.0
    Ti = 2 * L
    Td = 0.5 * L
    Ki = Kp / Ti if Ti != 0 else 0.0
    Kd = Kp * Td
    return Kp, Ki, Kd

def cria_pid_tf(Kp, Ki, Kd):
    """Cria uma função de transferência para um controlador PID."""
    s = control.TransferFunction.s
    return Kp + Ki / s + Kd * s

def malha_fechada_tf(Gp, Gc):
    """Calcula a função de transferência em malha fechada."""
    return control.feedback(Gp * Gc, 1)

def tabela_routh(coeficientes):
    """
    Gera a tabela de Routh para análise de estabilidade.
    Espera coeficientes numéricos.
    """
    # Garante que os coeficientes são uma lista plana de floats
    coeficientes = [float(c[0]) if isinstance(c, list) else float(c) for c in coeficientes]
    
    n = len(coeficientes)
    if n == 0:
        return np.array([])
    
    # O número de colunas na tabela de Routh
    m = (n + 1) // 2
    routh = np.zeros((n, m))
    
    # Preenche as duas primeiras linhas da tabela
    routh[0, :len(coeficientes[::2])] = coeficientes[::2]
    if n > 1:
        routh[1, :len(coeficientes[1::2])] = coeficientes[1::2]
    
    # Calcula as linhas subsequentes
    for i in range(2, n):
        # Verifica se o primeiro elemento da linha anterior é zero para evitar divisão por zero
        if routh[i - 1, 0] == 0:
            # Substitui por um pequeno valor para lidar com o caso de linha de zeros ou primeiro elemento zero
            # Isso é uma heurística comum para continuar a tabela, mas pode indicar raízes no eixo imaginário
            routh[i - 1, 0] = 1e-6 
        
        for j in range(m - 1):
            a = routh[i - 2, 0]
            b = routh[i - 1, 0]
            c = routh[i - 2, j + 1]
            d = routh[i - 1, j + 1]
            
            # Calcula o elemento da tabela
            routh[i, j] = (b * c - a * d) / b
    return routh

def salvar_grafico_resposta(t, y, nome, rotacao=0, deslocamento=0.0):
    """
    Salva um gráfico de resposta, com opções de rotação e deslocamento.
    """
    y = np.array(y) + deslocamento

    # Aplica rotações
    if rotacao == 180:
        t = -t[::-1]
        y = -y[::-1]
    elif rotacao == 90:
        t, y = -y, t # Troca e inverte eixos

    # Garante que os valores mínimos sejam não negativos para plotagem
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
    """Plota o diagrama de polos e zeros de uma função de transferência."""
    fig, ax = plt.subplots()
    ax.scatter(np.real(FT.poles()), np.imag(FT.poles()), marker='x', color='red', s=100, label='Polos')
    ax.scatter(np.real(FT.zeros()), np.imag(FT.zeros()), marker='o', color='blue', s=100, facecolors='none', edgecolors='blue', label='Zeros')
    
    # Desenha os eixos
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_title('Diagrama de Polos e Zeros')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_aspect('equal', adjustable='box') # Garante que os eixos tenham a mesma escala
    
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
            flash('Login de administrador bem-sucedido!')
            return redirect(url_for('admin'))

        if email in usuarios and usuarios[email]['senha'] == senha:
            if usuarios[email].get('aprovado', False):
                session['usuario_logado'] = email
                session['is_admin'] = False
                flash('Login bem-sucedido!')
                return redirect(url_for('painel'))
            else:
                flash('Seu cadastro ainda não foi aprovado. Por favor, aguarde a aprovação do administrador.')
                return redirect(url_for('login'))
        else:
            flash('Credenciais inválidas. Verifique seu email e senha.')
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
            flash('Este email já está cadastrado ou reservado. Por favor, use outro.')
            return redirect(url_for('cadastro'))

        usuarios[email] = {'nome': nome, 'senha': senha, 'aprovado': False}
        with open('usuarios.json', 'w') as f:
            json.dump(usuarios, f, indent=4)

        flash('Seu cadastro foi enviado para aprovação. Você será notificado por email quando for aprovado.')
        return redirect(url_for('login'))
    return render_template('cadastro.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    # Permite acesso somente se estiver logado como admin (email fixo + flag)
    if 'usuario_logado' not in session or session.get('is_admin') != True:
        flash('Acesso negado. Apenas o administrador pode acessar esta página.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        email_to_approve = request.form.get('email')
        if os.path.exists('usuarios.json'):
            with open('usuarios.json', 'r') as f:
                usuarios = json.load(f)
            if email_to_approve in usuarios:
                usuarios[email_to_approve]['aprovado'] = True
                with open('usuarios.json', 'w') as f:
                    json.dump(usuarios, f, indent=4)
                flash(f'Usuário {email_to_approve} aprovado com sucesso!')
            else:
                flash(f'Usuário {email_to_approve} não encontrado.')
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
        flash('Acesso negado. Por favor, faça login.')
        return redirect(url_for('login'))
    return render_template('painel.html')

@app.route('/logout')
def logout():
    session.pop('usuario_logado', None)
    session.pop('is_admin', None)
    flash('Você foi desconectado com sucesso.')
    return redirect(url_for('login'))

@app.route('/simulador', methods=['GET', 'POST'])
def simulador():
    if 'usuario_logado' not in session:
        flash('Faça login para acessar o simulador.')
        return redirect(url_for('login'))

    email = session['usuario_logado']

    # Verificar se usuário está aprovado (exceto para o admin fixo)
    if email != 'tisaaceng@gmail.com':
        if os.path.exists('usuarios.json'):
            with open('usuarios.json', 'r') as f:
                usuarios = json.load(f)
            if not usuarios.get(email, {}).get('aprovado', False):
                flash('Seu cadastro ainda não foi aprovado para usar o simulador. Por favor, aguarde a aprovação do administrador.')
                return redirect(url_for('painel'))
        else:
            flash('Arquivo de usuários não encontrado. Não foi possível verificar o status de aprovação.')
            return redirect(url_for('login'))

    resultado = None
    error = None
    warning = None # Nova variável para mensagens de aviso

    if request.method == 'POST':
        edo = request.form.get('edo')
        entrada = request.form.get('entrada')
        saida = request.form.get('saida')

        if not edo or not entrada or not saida:
            error = "Por favor, preencha todos os campos da Equação Diferencial Ordinária, Variável de Entrada e Variável de Saída."
        else:
            try:
                # Chama a função parse_edo, que agora retorna a FT numérica (ou None) e a flag simbólica
                Ls_expr, FT, has_symbolic_coeffs = parse_edo(edo, entrada, saida)
                ft_latex = ft_to_latex(Ls_expr)
                
                resultado = {
                    'ft_latex': ft_latex,
                    'is_symbolic': has_symbolic_coeffs # Passa esta flag para o template
                }

                if has_symbolic_coeffs:
                    warning = "A função de transferência contém coeficientes simbólicos. A análise numérica (resposta ao degrau, polos/zeros, sintonia PID, Tabela de Routh) não pode ser realizada. Por favor, forneça coeficientes numéricos para essas análises."
                else:
                    # Procede com a análise numérica apenas se não houver coeficientes simbólicos
                    t_open, y_open = resposta_degrau(FT)
                    L, T = estima_LT(t_open, y_open)
                    Kp, Ki, Kd = sintonia_ziegler_nichols(L, T)
                    pid = cria_pid_tf(Kp, Ki, Kd)
                    mf = malha_fechada_tf(FT, pid)
                    t_closed, y_closed = resposta_degrau(mf)

                    # Função auxiliar para converter TransferFunction do control para sympy.Expr
                    def tf_to_sympy_tf(tf_obj):
                        num = tf_obj.num[0][0]
                        den = tf_obj.den[0][0]
                        s_sym = sp.symbols('s') # Usa um símbolo 's' local
                        num_poly = sum(coef * s_sym**(len(num)-i-1) for i, coef in enumerate(num))
                        den_poly = sum(coef * s_sym**(len(den)-i-1) for i, coef in enumerate(den))
                        return num_poly / den_poly

                    expr_pid = sp.simplify(tf_to_sympy_tf(pid))
                    expr_mf = sp.simplify(tf_to_sympy_tf(mf))

                    # Salva os gráficos e obtém seus caminhos
                    img_resposta_aberta_path = salvar_grafico_resposta(t_open, y_open, 'resposta_malha_aberta', rotacao=180)
                    img_resposta_fechada_path = salvar_grafico_resposta(t_closed, y_closed, 'resposta_malha_fechada', rotacao=90, deslocamento=1.0)
                    img_pz_path = plot_polos_zeros(FT)

                    # Prepara a tabela de Routh
                    den_coefs = flatten_and_convert(FT.den[0])
                    routh_table = tabela_routh(den_coefs)

                    # Adiciona os resultados numéricos ao dicionário
                    resultado.update({
                        'pid_latex': sp.latex(expr_pid, mul_symbol='dot'),
                        'mf_latex': sp.latex(expr_mf, mul_symbol='dot'),
                        'Kp': Kp,
                        'Ki': Ki,
                        'Kd': Kd,
                        'img_resposta_aberta': img_resposta_aberta_path,
                        'img_resposta_fechada': img_resposta_fechada_path,
                        'img_pz': img_pz_path,
                        'routh_table': routh_table.tolist()
                    })

            except ValueError as ve: # Captura erros de validação ou parsing
                error = f"Erro de entrada ou processamento: {str(ve)}"
            except Exception as e: # Captura outros erros inesperados
                error = f"Ocorreu um erro inesperado: {str(e)}. Por favor, verifique a EDO e as variáveis."
    return render_template('simulador.html', resultado=resultado, error=error, warning=warning)

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
        
        if not nova_senha:
            flash('A nova senha não pode ser vazia.')
            return redirect(url_for('alterar_senha'))

        usuarios[email]['senha'] = nova_senha
        with open('usuarios.json', 'w') as f:
            json.dump(usuarios, f, indent=4)

        flash('Senha alterada com sucesso.')
        return redirect(url_for('perfil'))

    return render_template('alterar_senha.html')

@app.route('/funcao_transferencia')
def funcao_transferencia():
    # Esta rota pode ser usada para exibir a FT de forma isolada, se necessário.
    # Por enquanto, a FT é exibida diretamente no simulador.
    ft_latex = session.get('ft_latex', "Função de Transferência não disponível.")
    return render_template('transferencia.html', ft_latex=ft_latex)

# === EXECUÇÃO PRINCIPAL ===
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)
