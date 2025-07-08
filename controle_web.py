from flask import Flask, render_template, request, redirect, url_for, flash, session
import json
import os
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import control
from control.matlab import step
from sympy.abc import s # Importa a variável simbólica 's' globalmente para SymPy

# DEFININDO O CAMINHO RAIZ DA APLICAÇÃO DIRETAMENTE
# ESTE É O CAMINHO ONDE SUA PASTA 'Projeto2' DEVE ESTAR NO SEU CELULAR
# PARA QUE O PYDROID 3 TENHA ACESSO TOTAL DE ESCRITA.
BASE_DIR_FORCADO = '/storage/emulated/0/Android/data/ru.iiec.pydroid3/files/Projeto2'

app = Flask(__name__, root_path=BASE_DIR_FORCADO) # FORÇA O FLASK A USAR ESTE CAMINHO
app.secret_key = 'sua_chave_secreta' # Chave secreta para segurança da sessão do Flask

# REMOVIDO: A LINHA DEBUG ANTERIOR PARA EVITAR CONFUSÃO.
# Agora o caminho está sendo forçado, não apenas debugado.

# =============== FUNÇÕES AUXILIARES ===============

def flatten_and_convert(lst):
    """
    Achata uma lista possivelmente aninhada e converte todos os elementos para float.
    Lida com arrays NumPy também. Essencial para garantir formato correto dos coeficientes.
    """
    # Se a entrada é um array NumPy, achata e converte para lista Python.
    if isinstance(lst, np.ndarray):
        lst = lst.flatten().tolist()
    
    result = []
    for c in lst:
        # Se for um iterável (mas não string/bytes), chama recursivamente para achatar.
        if hasattr(c, '__iter__') and not isinstance(c, (str, bytes)):
            result.extend(flatten_and_convert(c))
        else:
            try:
                # Tenta converter o elemento para float.
                result.append(float(c))
            except Exception as e:
                # Levanta um erro se a conversão falhar.
                raise Exception(f"Erro convertendo coeficiente: {c} ({e})")
    return result

def pad_coeffs(num_coeffs, den_coeffs):
    """
    Preenche os coeficientes do numerador e denominador com zeros à esquerda
    para garantir que ambos tenham o mesmo comprimento, se necessário.
    Usado principalmente internamente em `parse_edo` para padronização.
    """
    # Garante que as entradas são listas planas de floats.
    num_coeffs = flatten_and_convert(num_coeffs)
    den_coeffs = flatten_and_convert(den_coeffs)

    len_num = len(num_coeffs)
    len_den = len(den_coeffs)
    min_len = 1 # Garante no mínimo 1 coeficiente (para polinômio constante).
    max_len = max(len_num, len_den, min_len) # Determina o maior comprimento.

    # Preenche com zeros à esquerda se o comprimento for menor que o máximo.
    if len_num < max_len:
        num_coeffs = [0.0] * (max_len - len_num) + num_coeffs
    if len_den < max_len:
        den_coeffs = [0.0] * (max_len - len_den) + den_coeffs
    
    return num_coeffs, den_coeffs

def remove_leading_zeros(coeffs):
    """
    Remove zeros à esquerda de uma lista de coeficientes, exceto se a lista for [0.0].
    Importante para a biblioteca `control` interpretar a ordem correta.
    """
    if not coeffs:
        return [0.0] # Retorna [0.0] se a lista estiver vazia.
    
    first_nonzero = 0
    # Encontra o índice do primeiro coeficiente não-zero (ou muito próximo de zero, considerando floats).
    for i, c in enumerate(coeffs):
        if abs(c) > 1e-9: # Usa uma pequena tolerância para comparação de floats com zero.
            first_nonzero = i
            break
    
    # Se todos os coeficientes são zero (ou muito próximos), retorna apenas [0.0]
    # para representar um polinômio nulo.
    if first_nonzero == len(coeffs) - 1 and abs(coeffs[first_nonzero]) < 1e-9:
         return [0.0]
    
    # Retorna a sub-lista a partir do primeiro coeficiente não-zero.
    return coeffs[first_nonzero:] if first_nonzero < len(coeffs) else [0.0]


def parse_edo(edo_str, entrada_str, saida_str):
    """
    Analisa uma Equação Diferencial Ordinária (EDO) em string,
    aplica a Transformada de Laplace para obter a Função de Transferência (FT).
    """
    t = sp.symbols('t', real=True) # Variável simbólica de tempo.
    # s já é importado globalmente de sympy.abc.s

    # Símbolos de Laplace para a saída (e.g., Y(s) ou X(s)) e entrada (e.g., U(s) ou F(s)).
    Xs = sp.Symbol(saida_str + '_s') 
    Fs = sp.Symbol(entrada_str + '_s') 

    # 1. Criar instâncias das funções simbólicas APLICADAS à variável 't' (e.g., y(t), u(t)).
    # Isso é crucial para que o SymPy as reconheça como funções que podem ser diferenciadas.
    y_func_applied = sp.Function(saida_str)(t) 
    u_func_applied = sp.Function(entrada_str)(t) 

    # 2. Definir as "funções base" (e.g., sp.Function('y'), sp.Function('u')).
    # Elas são usadas para comparar e identificar a qual função uma derivada ou termo pertence
    # (por exemplo, para distinguir diff(y,t,2) de diff(u,t,2)).
    y_func_base = sp.Function(saida_str) 
    u_func_base = sp.Function(entrada_str) 

    # 3. Preparar o `local_dict` para `sp.sympify`.
    # Mapeamos as STRINGS (e.g., 'y', 'u', 'diff') para os objetos SymPy correspondentes.
    # O mapeamento da string 'y' para o objeto y(t) (y_func_applied) é o que corrige o erro "UndefinedFunction".
    local_dict_sympify = {
        'sp': sp, 't': t, 's': s, # Inclui SymPy, t e s.
        saida_str: y_func_applied,   # Ex: 'y' (string) é mapeado para o objeto SymPy y(t).
        entrada_str: u_func_applied, # Ex: 'u' (string) é mapeado para o objeto SymPy u(t).
        'diff': sp.Derivative        # A string 'diff' é mapeada para o objeto sp.Derivative.
    }

    # Processar a string da EDO: converter para formato de expressão (LHS - RHS).
    if '=' in edo_str:
        lhs, rhs = edo_str.split('=')
        eq_str_expression = f"({lhs.strip()}) - ({rhs.strip()})"
    else:
        eq_str_expression = edo_str.strip()

    try:
        # Tenta converter a string da EDO para uma expressão SymPy.
        # Com o `local_dict_sympify` corrigido, SymPy deve parsear corretamente
        # as derivadas como sp.Derivative(y(t), t, 2), etc.
        edo_sym = sp.sympify(eq_str_expression, locals=local_dict_sympify)
    except (SyntaxError, ValueError) as e:
        # Captura erros de sintaxe ou valor e levanta uma exceção personalizada.
        raise ValueError(f"Erro de sintaxe na equação diferencial. Verifique o formato, ex: diff(y, t, 2) + 5*y = u. Detalhes: {e}")

    # === APLICANDO A TRANSFORMADA DE LAPLACE TERMO A TERMO ===
    laplace_expression = 0 # Inicializa a expressão transformada em Laplace.
    
    # Percorre cada termo da expressão SymPy (separando-os por soma/subtração).
    for arg in sp.Add.make_args(edo_sym):
        if isinstance(arg, sp.Derivative):
            # Se o termo é uma derivada (e.g., Derivative(y(t), t, 2))
            func_expr = arg.expr # Extrai a função que está sendo derivada (e.g., y(t))
            order = arg.derivative_count # Extrai a ordem da derivada (e.g., 2)
            
            # Compara a "função base" (o SymPy.Function 'y' ou 'u') da expressão derivada
            # para aplicar a transformada correta (Xs ou Fs).
            if func_expr.func == y_func_base: # Se for derivada de y(t)
                laplace_expression += (s**order) * Xs * (arg / sp.Derivative(y_func_applied,t,order)) # Multiplica pelo coeficiente se houver (ex: 2*diff(y,t,2))
            elif func_expr.func == u_func_base: # Se for derivada de u(t)
                laplace_expression += (s**order) * Fs * (arg / sp.Derivative(u_func_applied,t,order)) # Multiplica pelo coeficiente se houver
            else:
                # Se for uma derivada de outra função não mapeada, mantém o termo original.
                laplace_expression += arg
        else:
            # === CORREÇÃO CRÍTICA AQUI para termos como '4*y' ou '8*u' ===
            # Verificamos se o termo 'arg' contém a função de saída (y(t)) ou de entrada (u(t)).
            if y_func_applied in arg.atoms(sp.Function): 
                # Extrai o coeficiente de y(t) do termo (e.g., em '4*y(t)', extrai '4').
                coeff_of_y = arg.coeff(y_func_applied) 
                laplace_expression += coeff_of_y * Xs # Adiciona ao lado da saída.
            elif u_func_applied in arg.atoms(sp.Function): 
                # Extrai o coeficiente de u(t) do termo (e.g., em '8*u(t)', extrai '8').
                coeff_of_u = arg.coeff(u_func_applied)
                laplace_expression += coeff_of_u * Fs # Adiciona ao lado da entrada.
            else:
                # Se o termo não é uma derivada, nem contém y(t) ou u(t) (e.g., uma constante numérica).
                laplace_expression += arg # Mantém o termo original.

    laplace_expression = sp.simplify(laplace_expression) # Simplifica a expressão Laplace resultante.

    # === ISOLANDO A FUNÇÃO DE TRANSFERÊNCIA (FT = Saída(s) / Entrada(s)) ===
    
    # Extrai os coeficientes de Xs e Fs da expressão transformada.
    coef_Xs = laplace_expression.coeff(Xs)
    coef_Fs = laplace_expression.coeff(Fs)
    
    # Verifica se o coeficiente da saída é zero, o que impediria a formação da FT.
    if sp.simplify(coef_Xs) == 0: 
        raise ValueError(f"O coeficiente de '{saida_str}_s' (saída em Laplace) na transformada é zero. A função de transferência não pode ser determinada. Verifique a EDO.")

    # A Função de Transferência G(s) = X(s) / F(s) = - (coeficiente de Fs) / (coeficiente de Xs)
    Ls_expr = -coef_Fs / coef_Xs
    Ls_expr = sp.simplify(Ls_expr) # Simplifica a expressão final da FT.
    
    # Extrai o numerador e denominador da FT simplificada.
    num, den = sp.fraction(Ls_expr)
    
    # Converte numerador e denominador para polinômios de 's' e extrai seus coeficientes.
    num_poly = sp.Poly(num, s)
    den_poly = sp.Poly(den, s)
    
    num_coeffs = [float(c.evalf()) for c in num_poly.all_coeffs()]
    den_coeffs = [float(c.evalf()) for c in den_poly.all_coeffs()]

    # Garante que os coeficientes são listas planas de floats.
    num_coeffs = flatten_and_convert(num_coeffs)
    den_coeffs = flatten_and_convert(den_coeffs)
    
    # Remove zeros iniciais desnecessários (ex: [0, 0, 1, 2] -> [1, 2]) antes de criar a FT.
    num_coeffs_cleaned = remove_leading_zeros(num_coeffs)
    den_coeffs_cleaned = remove_leading_zeros(den_coeffs)

    # Garante que as listas de coeficientes não estão vazias após a limpeza.
    if not num_coeffs_cleaned:
        num_coeffs_cleaned = [0.0]
    if not den_coeffs_cleaned:
        den_coeffs_cleaned = [0.0]

    # Cria o objeto control.TransferFunction da biblioteca python-control.
    FT = control.TransferFunction(num_coeffs_cleaned, den_coeffs_cleaned)
    
    return Ls_expr, FT

def ft_to_latex(expr):
    """Converte uma expressão SymPy para formato LaTeX."""
    return sp.latex(expr, mul_symbol='dot')

def resposta_degrau(FT, tempo=None):
    """
    Calcula e retorna a resposta ao degrau de uma Função de Transferência.
    Ajusta o tempo de simulação automaticamente se não for fornecido.
    """
    if tempo is None:
        poles = FT.poles()
        if poles.size > 0:
            # Estima a constante de tempo do sistema a partir do polo dominante (real com menor magnitude).
            max_abs_real_pole = np.max(np.abs(np.real(poles)))
            if max_abs_real_pole > 0:
                estimated_time_constant = 1.0 / (max_abs_real_pole + 1e-9) 
                # Tempo final para acomodação (5 vezes a constante de tempo), limitado a 100s.
                end_time = min(5.0 * estimated_time_constant * 5, 100.0) 
                if end_time < 5.0: # Garante um mínimo de 5 segundos para sistemas muito rápidos.
                    end_time = 5.0
                tempo = np.linspace(0, end_time, 1000)
            else: # Se os polos estão em zero ou no eixo imaginário (oscilatório não estável).
                tempo = np.linspace(0, 10, 1000)
        else: # Para sistemas sem polos (ganho puro ou constante).
            tempo = np.linspace(0, 10, 1000)

    t, y = step(FT, T=tempo)
    
    # Garante que t e y são arrays NumPy planos.
    t = np.array(t).flatten()
    y = np.array(y).flatten()
    
    return t, y

def estima_LT(t, y):
    """
    Estima o tempo morto (L) e a constante de tempo (T) da resposta ao degrau
    usando o método do ponto de inflexão ou similar (para modelo de primeira ordem com atraso).
    """
    if len(y) == 0:
        return 0.0, 0.0 # Retorna zero se a resposta estiver vazia.

    y_final = y[-1] # Valor final da resposta ao degrau.
    
    # Se a saída é essencialmente zero ou o sistema é instável (não se acomoda).
    if abs(y_final) < 1e-6 and np.max(np.abs(y)) < 1e-6:
        return 0.0, 0.0 # Retorna zero se a resposta for nula ou insignificante.
    
    # Normaliza a resposta para o valor final (evita problemas com escala em valores muito pequenos/grandes).
    y_norm = y / (y_final + 1e-9) if abs(y_final) > 1e-9 else y

    # Estima L (tempo morto): encontra o primeiro ponto onde a resposta excede 1% do valor final.
    indice_inicio = next((i for i, val in enumerate(y_norm) if val > 0.01), 0)
    L = t[indice_inicio] if indice_inicio < len(t) else 0.0

    # Estima T (constante de tempo): encontra o ponto onde a resposta atinge 63.2% do valor final (após o atraso L).
    y_63_norm = 0.632 
    indice_63 = next((i for i, val in enumerate(y_norm) if val >= y_63_norm), len(y_norm) - 1)
    
    T = (t[indice_63] - L) if indice_63 < len(t) else 0.0

    # Garante que L e T sejam positivos e com um valor mínimo razoável para evitar divisões por zero ou valores irrealistas.
    L = max(L, 0.001) if L < 0.001 else L 
    T = max(T, 0.001) if T < 0.001 else T 
    
    return L, T

def sintonia_ziegler_nichols(L, T):
    """
    Calcula os parâmetros Kp, Ki, Kd para um controlador PID
    usando a tabela de Ziegler-Nichols (método da curva de reação).
    """
    # Lida com casos onde L ou T são zero/negativos para evitar divisões por zero.
    if L <= 0: 
        L = 0.001 # Assume um pequeno tempo morto para permitir cálculo.
    if T <= 0: 
        T = 0.001 # Assume uma pequena constante de tempo.

    Kp = 1.2 * T / L
    Ti = 2 * L
    Td = 0.5 * L
    
    Ki = Kp / Ti if Ti != 0 else 0.0 # Evita divisão por zero se Ti for 0.
    Kd = Kp * Td
    
    return Kp, Ki, Kd

def cria_pid_tf(Kp, Ki, Kd):
    """Cria a Função de Transferência de um controlador PID."""
    s = control.TransferFunction.s # Variável de Laplace para a biblioteca control.
    # Garante que os coeficientes são floats.
    Kp = float(Kp)
    Ki = float(Ki)
    Kd = float(Kd)
    return Kp + Ki / s + Kd * s

def malha_fechada_tf(Gp, Gc):
    """
    Calcula a Função de Transferência de malha fechada para um sistema com feedback unitário.
    Gp: Função de Transferência do Processo
    Gc: Função de Transferência do Controlador
    """
    return control.feedback(Gp * Gc, 1)

def tabela_routh(coef):
    """
    Gera a tabela de Routh para análise de estabilidade de um polinômio.
    Retorna a tabela como um array NumPy.
    Lida com casos especiais de zeros na primeira coluna de forma simplificada.
    """
    coef = flatten_and_convert(coef) # Garante que os coeficientes são uma lista plana de floats.

    n = len(coef) # Número de coeficientes (grau do polinômio + 1).
    if n == 0:
        return np.array([[]]) # Tabela vazia para nenhum coeficiente.
    if n == 1: # Polinômio de grau 0 (apenas uma constante).
        return np.array([[coef[0]]]) # Tabela com um único elemento.

    n_routh_rows = n # Número de linhas na tabela de Routh.
    m = (n + 1) // 2 # Número de colunas na tabela de Routh (arredondado para cima).

    routh = np.zeros((n_routh_rows, m)) # Inicializa a tabela com zeros.

    # Preenche a primeira linha (coeficientes de potências pares de s).
    first_row_coeffs = np.array(coef[::2])
    routh[0, :len(first_row_coeffs)] = first_row_coeffs

    # Preenche a segunda linha (coeficientes de potências ímpares de s), se houver.
    if n > 1: 
        second_row_coeffs = np.array(coef[1::2])
        routh[1, :len(second_row_coeffs)] = second_row_coeffs
    
    # Preenche as linhas restantes da tabela de Routh.
    for i in range(2, n):
        # Lida com o caso de zero (ou valor muito próximo de zero) no primeiro elemento da linha anterior
        # Substitui por um pequeno épsilon para evitar divisão por zero, o que pode indicar estabilidade marginal.
        if abs(routh[i - 1, 0]) < 1e-9: 
            routh[i - 1, 0] = 1e-9 
            
        for j in range(m): # Itera sobre todas as colunas.
            a = routh[i - 2, 0] # Elemento da linha i-2, coluna 0 (o pivô superior).
            b = routh[i - 1, 0] # Elemento da linha i-1, coluna 0 (o pivô da linha anterior).
            
            # Elementos nas colunas j+1 das linhas anteriores (com verificação de limite para evitar IndexError).
            c = routh[i - 2, j + 1] if (j + 1) < m else 0.0
            d = routh[i - 1, j + 1] if (j + 1) < m else 0.0
            
            # Cálculo do elemento atual da tabela de Routh.
            if abs(b) < 1e-9: # Se o pivô da linha anterior é zero (ou muito próximo).
                routh[i, j] = 0.0 # Define como 0.0 para evitar divisão por zero.
            else:
                routh[i, j] = (b * c - a * d) / b
                
        return routh

    def salvar_grafico_resposta(t, y, nome, rotacao=0, deslocamento=0.0):
        """
        Salva um gráfico de resposta de tempo.
        Remove valores NaN/Inf e aplica transformações básicas de rotação/deslocamento.
        """
        # Filtra valores inválidos (NaNs e Infs) dos dados de tempo e saída.
        valid_indices = ~np.isnan(t) & ~np.isinf(t) & ~np.isinf(y)
        t = t[valid_indices]
        y = y[valid_indices]

        if len(t) == 0 or len(y) == 0:
            print(f"Não há dados válidos para o gráfico {nome}. Pulando.")
            return None # Retorna None se não houver dados válidos para plotar.

        y = y + deslocamento # Aplica um deslocamento vertical à saída.
        
        # Aplica rotações (geralmente não usado para gráficos de resposta de tempo padrão, mas mantido).
        if rotacao == 180:
            t, y = -t[::-1], -y[::-1]
        elif rotacao == 90:
            t_temp = t.copy()
            t = y
            y = t_temp

        # Ajusta os eixos para que comecem em zero, se houver valores negativos (para melhor visualização).
        if t.size > 0 and np.min(t) < 0: t = t - np.min(t)
        if y.size > 0 and np.min(y) < 0: y = y - np.min(y)

        plt.figure(figsize=(8, 4)) # Cria uma nova figura para o gráfico.
        plt.plot(t, y, label='Resposta ao Degrau') # Plota a resposta.
        plt.xlabel('Tempo (s)' if rotacao != 90 else 'Saída') # Rótulo do eixo X.
        plt.ylabel('Saída' if rotacao != 90 else 'Tempo (s)') # Rótulo do eixo Y.
        plt.title(nome) # Título do gráfico.
        plt.grid(True) # Adiciona grade.
        plt.legend() # Adiciona legenda.
        plt.tight_layout() # Ajusta o layout para evitar sobreposição de elementos.

        caminho = os.path.join('static', f'{nome}.png') # Define o caminho para salvar a imagem.
        plt.savefig(caminho) # Salva a figura como arquivo PNG.
        plt.close() # Fecha a figura para liberar memória (importante em servidores).
        return caminho # Retorna o caminho do arquivo salvo.

    def plot_polos_zeros(FT):
        """
        Plota o diagrama de polos e zeros para uma Função de Transferência.
        """
        if not isinstance(FT, control.TransferFunction):
            print("Objeto FT inválido para plotar polos e zeros.")
            return None

        poles = FT.poles() # Obtém os polos da FT.
        zeros = FT.zeros() # Obtém os zeros da FT.

        if len(poles) == 0 and len(zeros) == 0:
            print("Não há polos ou zeros para plotar.")
            return None # Retorna None se não houver polos nem zeros.

        fig, ax = plt.subplots() # Cria uma nova figura e eixos.
        
        # Plota os polos (marcadores 'x' vermelhos).
        if poles.size > 0:
            ax.scatter(np.real(poles), np.imag(poles), marker='x', color='red', label='Polos', s=100)
        # Plota os zeros (marcadores 'o' azuis com face vazia).
        if zeros.size > 0:
            ax.scatter(np.real(zeros), np.imag(zeros), marker='o', color='blue', label='Zeros', s=100, facecolors='none', edgecolors='blue') 

        # Desenha os eixos real (horizontal) e imaginário (vertical).
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Re') # Rótulo do eixo real.
        ax.set_ylabel('Im') # Rótulo do eixo imaginário.
        ax.set_title('Diagrama de Polos e Zeros') # Título do gráfico.
        ax.legend() # Adiciona legenda.
        ax.grid(True) # Adiciona grade.
        ax.set_aspect('equal', adjustable='box') # Garante que os eixos tenham a mesma escala para visualização correta.

        # Ajusta os limites dos eixos para incluir todos os polos e zeros com uma margem.
        all_points = np.concatenate((poles, zeros))
        if all_points.size > 0:
            max_coord = np.max(np.abs(all_points))
            margin = max_coord * 0.1 if max_coord > 0 else 1.0 # Adiciona uma margem de 10% ou 1.0.
            ax.set_xlim([-max_coord - margin, max_coord + margin])
            ax.set_ylim([-max_coord - margin, max_coord + margin])

        caminho = os.path.join('static', 'polos_zeros.png') # Define o caminho para salvar a imagem.
        plt.savefig(caminho) # Salva a figura como arquivo PNG.
        plt.close() # Fecha a figura para liberar memória.
        return caminho # Retorna o caminho do arquivo salvo.

    # ================ ROTAS DO FLASK ====================

    @app.route('/')
    def home():
        """Rota da página inicial."""
        admin = False
        if 'usuario_logado' in session:
            # Verifica se o arquivo de usuários existe e carrega os dados.
            if os.path.exists('usuarios.json'):
                with open('usuarios.json', 'r') as f:
                    usuarios = json.load(f)
                email = session['usuario_logado']
                admin = usuarios.get(email, {}).get('admin', False) # Verifica se o usuário logado é admin.
        return render_template('index.html', admin=admin)

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """Rota para login de usuários."""
        if request.method == 'POST':
            email = request.form['email']
            senha = request.form['senha']
            # Verifica se o arquivo de usuários existe.
            if os.path.exists('usuarios.json'):
                with open('usuarios.json', 'r') as f:
                    usuarios = json.load(f)
            else:
                usuarios = {} # Se não existe, inicializa como vazio.
            
            # Verifica credenciais e status de aprovação.
            if email in usuarios and usuarios[email]['senha'] == senha:
                if usuarios[email].get('aprovado', False):
                    session['usuario_logado'] = email # Define a sessão do usuário.
                    flash('Login realizado com sucesso!')
                    return redirect(url_for('painel')) # Redireciona para o painel.
                else:
                    flash('Cadastro ainda não aprovado. Aguarde a aprovação do administrador.')
                    return redirect(url_for('login'))
            else:
                flash('Credenciais inválidas. Verifique seu e-mail e senha.')
        return render_template('login.html')

    @app.route('/cadastro', methods=['GET', 'POST'])
    def cadastro():
        """Rota para registro de novos usuários."""
        if request.method == 'POST':
            nome = request.form['nome']
            email = request.form['email']
            senha = request.form['senha']
            # Carrega usuários existentes ou inicializa.
            if os.path.exists('usuarios.json'):
                with open('usuarios.json', 'r') as f:
                    usuarios = json.load(f)
            else:
                usuarios = {}
            
            # Verifica se o e-mail já está cadastrado.
            if email in usuarios:
                flash('Este e-mail já está cadastrado. Tente fazer login ou use outro e-mail.')
                return redirect(url_for('cadastro'))
            
            # Adiciona novo usuário com status 'aprovado': False.
            usuarios[email] = {'nome': nome, 'senha': senha, 'aprovado': False}
            with open('usuarios.json', 'w') as f:
                json.dump(usuarios, f, indent=4) # Salva os usuários.
            flash('Cadastro enviado para aprovação. Você será notificado quando for aprovado.')
            return redirect(url_for('login'))
        return render_template('cadastro.html')

    @app.route('/admin', methods=['GET', 'POST'])
    def admin():
        """Rota para a página de administração (aprovação de usuários)."""
        if 'usuario_logado' not in session:
            flash('Acesso negado. Faça login para acessar esta página.')
            return redirect(url_for('login'))
        
        # Carrega usuários existentes.
        if os.path.exists('usuarios.json'):
            with open('usuarios.json', 'r') as f:
                usuarios = json.load(f)
            # A partir daqui, se o arquivo não existir, o Flask retornará um 500 ou outro erro.
            # O tratamento de erro no simulador não é para este caso.
        else:
            usuarios = {}

        email = session['usuario_logado']
        # Verifica se o usuário logado tem permissão de admin.
        if not usuarios.get(email, {}).get('admin', False):
            flash('Acesso restrito ao administrador.')
            return redirect(url_for('painel'))
        
        if request.method == 'POST':
            email_aprovar = request.form.get('email')
            if email_aprovar in usuarios:
                usuarios[email_aprovar]['aprovado'] = True # Aprova o usuário.
                with open('usuarios.json', 'w') as f:
                    json.dump(usuarios, f, indent=4)
                flash(f'Usuário {email_aprovar} aprovado com sucesso!')
            else:
                flash(f'Usuário {email_aprovar} não encontrado.')

        # Filtra usuários não aprovados para exibição.
        nao_aprovados = {k: v for k, v in usuarios.items() if not v.get('aprovado', False)}
        return render_template('admin.html', usuarios=nao_aprovados)

    @app.route('/painel')
    def painel():
        """Rota do painel de controle para usuários logados."""
        if 'usuario_logado' not in session:
            flash('Acesso negado. Faça login para acessar o painel.')
            return redirect(url_for('login'))
        return render_template('painel.html')

    @app.route('/perfil')
    def perfil():
        """Rota da página de perfil do usuário."""
        if 'usuario_logado' not in session:
            flash('Faça login para acessar o perfil.')
            return redirect(url_for('login'))
        email = session['usuario_logado']
        if os.path.exists('usuarios.json'):
            with open('usuarios.json', 'r') as f:
                usuarios = json.load(f)
            # A partir daqui, se o arquivo não existir, o Flask retornará um 500 ou outro erro.
            # O tratamento de erro no simulador não é para este caso.
        else:
            usuarios = {}
        usuario = usuarios.get(email) # Obtém dados do usuário logado.
        return render_template('perfil.html', usuario=usuario, email=email)

    @app.route('/alterar_senha', methods=['GET', 'POST'])
    def alterar_senha():
        """Rota para alteração de senha do usuário."""
        if 'usuario_logado' not in session:
            flash('Faça login para alterar a senha.')
            return redirect(url_for('login'))
        if request.method == 'POST':
            senha_atual = request.form['senha_atual']
            nova_senha = request.form['nova_senha']
            confirmar = request.form['confirmar_senha']
            email = session['usuario_logado']
            
            if os.path.exists('usuarios.json'):
                with open('usuarios.json', 'r') as f:
                    usuarios = json.load(f)
                # A partir daqui, se o arquivo não existir, o Flask retornará um 500 ou outro erro.
                # O tratamento de erro no simulador não é para este caso.
            else:
                # Caso o arquivo usuarios.json não exista ao tentar alterar a senha.
                flash('Erro: Arquivo de usuários não encontrado.')
                return redirect(url_for('perfil'))
                
            if usuarios[email]['senha'] != senha_atual:
                flash('Senha atual incorreta.')
                return redirect(url_for('alterar_senha'))
            if nova_senha != confirmar:
                flash('A nova senha e a confirmação não coincidem.')
                return redirect(url_for('alterar_senha'))
            
            usuarios[email]['senha'] = nova_senha # Atualiza a senha.
            with open('usuarios.json', 'w') as f:
                json.dump(usuarios, f, indent=4)
            flash('Senha alterada com sucesso.')
            return redirect(url_for('perfil'))
        return render_template('alterar_senha.html')

    @app.route('/logout')
    def logout():
        """Rota para logout do usuário."""
        session.pop('usuario_logado', None) # Remove o usuário da sessão.
        flash('Logout realizado com sucesso.')
        return redirect(url_for('login'))

    @app.route('/simulador', methods=['GET', 'POST'])
    def simulador():
        """Rota principal do simulador de sistemas de controle."""
        resultado = None # Inicializa resultado como None.
        error = None # Inicializa error como None.

        if 'usuario_logado' not in session:
            flash('Acesso restrito. Faça login para usar o simulador.')
            return redirect(url_for('login'))
        
        if request.method == 'POST':
            edo = request.form.get('edo')
            entrada = request.form.get('entrada')
            saida = request.form.get('saida')
            
            try:
                # 1. Garante que o diretório 'static' existe e é gravável.
                static_dir = os.path.join(app.root_path, 'static')
                if not os.path.exists(static_dir):
                    os.makedirs(static_dir)
                elif not os.access(static_dir, os.W_OK):
                    # Se não tem permissão de escrita, levanta um IOError.
                    raise IOError(f"O diretório 'static' não tem permissões de escrita. Caminho: {static_dir}")

                # 2. Analisa a EDO e obtém a Função de Transferência.
                Ls_expr, FT = parse_edo(edo, entrada, saida)
                ft_latex = ft_to_latex(Ls_expr)
                
                # 3. Calcula e salva o gráfico da Resposta ao Degrau em Malha Aberta.
                t_open, y_open = resposta_degrau(FT)
                img_aberta_path = salvar_grafico_resposta(t_open, y_open, 'resposta_malha_aberta', 0) 
                
                # 4. Estima os parâmetros L (tempo morto) e T (constante de tempo) para sintonia PID.
                L, T = estima_LT(t_open, y_open)

                # 5. Sintonia do controlador PID usando Ziegler-Nichols.
                Kp, Ki, Kd = sintonia_ziegler_nichols(L, T)
                pid = cria_pid_tf(Kp, Ki, Kd)
                
                # 6. Calcula a Função de Transferência em Malha Fechada com o PID.
                mf = malha_fechada_tf(FT, pid)
                t_closed, y_closed = resposta_degrau(mf)
                img_fechada_path = salvar_grafico_resposta(t_closed, y_closed, 'resposta_malha_fechada', 0, 0.0) 
                
                # 7. Prepara as expressões LaTeX para o PID e a Malha Fechada.
                # Extrai os coeficientes dos objetos TransferFunction e reconstrói as expressões SymPy.
                num_pid_coeffs = control.tfdata(pid)[0][0][0] 
                den_pid_coeffs = control.tfdata(pid)[1][0][0] 
                num_pid_poly = sum(c * s**(len(num_pid_coeffs) - 1 - i) for i, c in enumerate(num_pid_coeffs))
                den_pid_poly = sum(c * s**(len(den_pid_coeffs) - 1 - i) for i, c in enumerate(den_pid_coeffs))
                expr_pid = num_pid_poly / den_pid_poly if den_pid_poly != 0 else num_pid_poly 

                num_mf_coeffs = control.tfdata(mf)[0][0][0]
                den_mf_coeffs = control.tfdata(mf)[1][0][0]
                num_mf_poly = sum(c * s**(len(num_mf_coeffs) - 1 - i) for i, c in enumerate(num_mf_coeffs))
                den_mf_poly = sum(c * s**(len(den_mf_coeffs) - 1 - i) for i, c in enumerate(den_mf_coeffs))
                expr_mf = num_mf_poly / den_mf_poly if den_mf_poly != 0 else num_mf_poly

                # 8. Plota o Diagrama de Polos e Zeros.
                img_pz_path = plot_polos_zeros(FT)
                
                # 9. Gera a Tabela de Routh para análise de estabilidade.
                den_coeffs_for_routh = flatten_and_convert(FT.den[0])
                routh_table = tabela_routh(den_coeffs_for_routh)

                # 10. Prepara o dicionário 'resultado' para ser passado ao template.
                resultado = {
                    'ft_latex': ft_latex,
                    'pid_latex': sp.latex(sp.simplify(expr_pid), mul_symbol='dot'), 
                    'mf_latex': sp.latex(sp.simplify(expr_mf), mul_symbol='dot'),   
                    'Kp': Kp, 'Ki': Ki, 'Kd': Kd,
                    'img_resposta_aberta': img_aberta_path,
                    'img_resposta_fechada': img_fechada_path,
                    'img_pz': img_pz_path,
                    'routh_table': routh_table.tolist() # Converte para lista para ser serializável no template.
                }
            except Exception as e:
                # Captura qualquer erro e o exibe no frontend.
                error = f"Erro no simulador: {str(e)}. Por favor, verifique a EDO e as permissões."
        return render_template('simulador.html', resultado=resultado, error=error)

    # =============== EXECUÇÃO DA APLICAÇÃO FLASK ===============
    if __name__ == '__main__':
        # Define a porta, usando a variável de ambiente PORT se disponível, senão 5050.
        port = int(os.environ.get('PORT', 5050))
        # Inicia o servidor Flask.
        app.run(host='0.0.0.0', port=port, debug=True) # debug=True é para desenvolvimento.
