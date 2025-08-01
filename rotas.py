import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash, session

from funcoes_auxiliares import (
    parse_edo, ft_to_latex, resposta_degrau, estima_LT, sintonia_ziegler_nichols,
    cria_pid_tf, malha_fechada_tf, salvar_grafico_resposta, plot_polos_zeros,
    flatten_and_convert, tabela_routh
)

app = Flask(__name__)
app.secret_key = 'sua_chave_secreta'  # Altere para uma chave segura

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
                flash('Seu cadastro ainda não foi aprovado para usar o simulador.')
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
                        import sympy as sp
                        num = tf_obj.num[0][0]
                        den = tf_obj.den[0][0]
                        s_sym = sp.symbols('s') # Usa um símbolo 's' local
                        num_poly = sum(coef * s_sym**(len(num)-i-1) for i, coef in enumerate(num))
                        den_poly = sum(coef * s_sym**(len(den)-i-1) for i, coef in enumerate(den))
                        return num_poly / den_poly

                    import sympy as sp
                    expr_pid = sp.simplify(tf_to_sympy_tf(pid))
                    expr_mf = sp.simplify(tf_to_sympy_tf(mf))

                    # Salva os gráficos e obtém seus caminhos
                    img_resposta_aberta_path = salvar_grafico_resposta(t_open, y_open, 'resposta_malha_aberta', deslocamento=0.0)
                    img_resposta_fechada_path = salvar_grafico_resposta(t_closed, y_closed, 'resposta_malha_fechada', deslocamento=1.0)
                    img_pz_path = plot_polos_zeros(FT)

                    # Prepara a tabela de Routh
                    den_coefs = flatten_and_convert(FT.den[0])
                    routh_table = tabela_routh(den_coefs)

                    # Adiciona os resultados numéricos ao dicionário
                    resultado.update({
                        'pid_latex': sp.latex(expr_pid, mul_symbol='dot'),
                        'mf_latex': sp.latex(expr_mf, mul_symbol='dot'),
                        'Kp': round(Kp, 4),
                        'Ki': round(Ki, 4),
                        'Kd': round(Kd, 4),
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
