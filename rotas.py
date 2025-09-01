import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

from funcoes_auxiliares import (
    parse_edo, ft_to_latex, resposta_degrau, estima_LT, sintonia_ziegler_nichols,
    sintonia_oscilacao_forcada, cria_pid_tf, malha_fechada_tf, salvar_grafico_resposta,
    plot_polos_zeros, flatten_and_convert, tabela_routh, calcula_kc_tc_automaticamente
)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'sua_chave_secreta_padrao')

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    senha = db.Column(db.String(255), nullable=False)
    aprovado = db.Column(db.Boolean, default=False, nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)

    def __repr__(self):
        return f'<User {self.email}>'

with app.app_context():
    db.create_all()
    admin_email = 'tisaaceng@gmail.com'
    admin_user = User.query.filter_by(email=admin_email).first()
    if not admin_user:
        hashed_password = generate_password_hash('4839AT81', method='pbkdf2:sha256')
        new_admin = User(
            nome='Tiago Carneiro',
            email=admin_email,
            senha=hashed_password,
            aprovado=True,
            is_admin=True
        )
        db.session.add(new_admin)
        db.session.commit()
        print("Usuário administrador adicionado ao banco de dados.")

@app.route('/')
def home():
    user_email = session.get('usuario_logado')
    return render_template('index.html', user_email=user_email, is_admin=session.get('is_admin', False))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        senha = request.form['senha']
        
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.senha, senha):
            if user.aprovado:
                session['usuario_logado'] = user.email
                session['is_admin'] = user.is_admin
                flash('Login bem-sucedido!', 'success')
                if user.is_admin:
                    return redirect(url_for('admin'))
                else:
                    return redirect(url_for('painel'))
            else:
                flash('Seu cadastro ainda não foi aprovado. Por favor, aguarde a aprovação do administrador.', 'warning')
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
        
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Este email já está cadastrado. Por favor, use outro.', 'warning')
            return redirect(url_for('cadastro'))

        hashed_password = generate_password_hash(senha, method='pbkdf2:sha256')
        
        new_user = User(
            nome=nome,
            email=email,
            senha=hashed_password,
            aprovado=False,
            is_admin=False
        )
        db.session.add(new_user)
        db.session.commit()
        
        flash('Seu cadastro foi enviado para aprovação.', 'success')
        return redirect(url_for('login'))
    return render_template('cadastro.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if 'usuario_logado' not in session or not session.get('is_admin'):
        flash('Acesso negado. Apenas o administrador pode acessar esta página.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        email_to_process = request.form.get('email')
        action = request.form.get('action')
        user = User.query.filter_by(email=email_to_process).first()
        
        if user:
            if action == 'aprovar':
                user.aprovado = True
                db.session.commit()
                flash(f'Usuário {user.email} aprovado com sucesso!', 'success')
            elif action == 'excluir':
                if user.is_admin:
                    flash('Você não pode excluir a sua própria conta de administrador.', 'danger')
                else:
                    db.session.delete(user)
                    db.session.commit()
                    flash(f'Usuário {user.email} excluído com sucesso.', 'success')
        else:
            flash(f'Usuário {email_to_process} não encontrado.', 'warning')

    nao_aprovados = User.query.filter_by(aprovado=False, is_admin=False).all()
    aprovados = User.query.filter_by(aprovado=True, is_admin=False).all()
    
    return render_template('admin.html', nao_aprovados=nao_aprovados, aprovados=aprovados, is_admin=session.get('is_admin'))

@app.route('/painel')
def painel():
    if 'usuario_logado' not in session:
        flash('Acesso negado. Por favor, faça login.', 'danger')
        return redirect(url_for('login'))
    
    user = User.query.filter_by(email=session['usuario_logado']).first()
    is_admin = user.is_admin if user else False
    
    return render_template('painel.html', is_admin=is_admin)

@app.route('/logout')
def logout():
    session.pop('usuario_logado', None)
    session.pop('is_admin', None)
    flash('Você foi desconectado com sucesso.', 'success')
    return redirect(url_for('login'))

@app.route('/simulador', methods=['GET', 'POST'])
def simulador():
    if 'usuario_logado' not in session:
        flash('Faça login para acessar o simulador.', 'warning')
        return redirect(url_for('login'))

    user = User.query.filter_by(email=session['usuario_logado']).first()
    if not user or not user.aprovado:
        flash('Seu cadastro ainda não foi aprovado para usar o simulador.', 'warning')
        return redirect(url_for('painel'))

    resultado = None
    error = None
    warning = None

    if request.method == 'POST':
        session.pop('resultado', None)

        edo = request.form.get('edo')
        entrada = request.form.get('entrada')
        saida = request.form.get('saida')
        metodo_sintonia = request.form.get('metodo_sintonia')
        kc = request.form.get('kc')
        tc = request.form.get('tc')

        if not edo or not entrada or not saida:
            error = "Por favor, preencha todos os campos da Equação Diferencial Ordinária, Variável de Entrada e Variável de Saída."
        else:
            try:
                Ls_expr, FT, has_symbolic_coeffs = parse_edo(edo, entrada, saida)
                ft_latex = ft_to_latex(Ls_expr)
                
                resultado = {
                    'ft_latex': ft_latex,
                    'is_symbolic': has_symbolic_coeffs,
                    'method': metodo_sintonia
                }

                if has_symbolic_coeffs:
                    warning = "A função de transferência contém coeficientes simbólicos. A análise numérica (resposta ao degrau, polos/zeros, sintonia PID, Tabela de Routh) não pode ser realizada. Por favor, forneça coeficientes numéricos para essas análises."
                else:
                    Kp, Ki, Kd = 0, 0, 0
                    img_resposta_aberta_path = None
                    
                    if metodo_sintonia == 'degrau':
                        t_open, y_open = resposta_degrau(FT)
                        L, T = estima_LT(t_open, y_open)
                        Kp, Ki, Kd = sintonia_ziegler_nichols(L, T)
                        img_resposta_aberta_path = salvar_grafico_resposta(t_open, y_open, 'resposta_malha_aberta', deslocamento=0.0)
                    elif metodo_sintonia == 'oscilacao':
                        if not kc or not tc:
                            # Caso de cálculo automático
                            kc_auto, tc_auto = calcula_kc_tc_automaticamente(FT.den[0])
                            if kc_auto is None or tc_auto is None:
                                error = "Não foi possível calcular Kc e Tc automaticamente. O sistema pode ser instável ou a EDO não tem a forma esperada."
                                resultado = None
                                session['error'] = error
                                return redirect(url_for('simulador'))
                            kc_final = kc_auto
                            tc_final = tc_auto
                            warning = "Kc e Tc foram calculados automaticamente."
                        else:
                            # Caso de inserção manual
                            kc_final = float(kc)
                            tc_final = float(tc)

                        Kp, Ki, Kd = sintonia_oscilacao_forcada(kc_final, tc_final)
                    else:
                        raise ValueError("Método de sintonia inválido.")
                        
                    pid = cria_pid_tf(Kp, Ki, Kd)
                    mf = malha_fechada_tf(FT, pid)
                    t_closed, y_closed = resposta_degrau(mf)

                    def tf_to_sympy_tf(tf_obj):
                        import sympy as sp
                        num = tf_obj.num[0][0]
                        den = tf_obj.den[0][0]
                        s_sym = sp.symbols('s')
                        num_poly = sum(coef * s_sym**(len(num)-i-1) for i, coef in enumerate(num))
                        den_poly = sum(coef * s_sym**(len(den)-i-1) for i, coef in enumerate(den))
                        return num_poly / den_poly

                    import sympy as sp
                    expr_pid = sp.simplify(tf_to_sympy_tf(pid))
                    expr_mf = sp.simplify(tf_to_sympy_tf(mf))

                    img_resposta_fechada_path = salvar_grafico_resposta(t_closed, y_closed, 'resposta_malha_fechada', deslocamento=1.0)
                    img_pz_path = plot_polos_zeros(FT)
                    
                    den_coefs = flatten_and_convert(FT.den[0])
                    routh_table = tabela_routh(den_coefs)

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

            except ValueError as ve:
                error = f"Erro de entrada ou processamento: {str(ve)}"
            except Exception as e:
                error = f"Ocorreu um erro inesperado: {str(e)}. Por favor, verifique a EDO e as variáveis."
        
        session['resultado'] = resultado
        session['error'] = error
        session['warning'] = warning
        return redirect(url_for('simulador'))

    resultado_da_sessao = session.get('resultado', None)
    error_da_sessao = session.get('error', None)
    warning_da_sessao = session.get('warning', None)
    is_admin = session.get('is_admin', False)

    return render_template(
        'simulador.html',
        resultado=resultado_da_sessao,
        error=error_da_sessao,
        warning=warning_da_sessao,
        is_admin=is_admin
    )
    
@app.route('/perfil')
def perfil():
    if 'usuario_logado' not in session:
        flash('Faça login para acessar o perfil.', 'warning')
        return redirect(url_for('login'))

    user = User.query.filter_by(email=session['usuario_logado']).first()
    if user:
        is_admin = user.is_admin
        return render_template('perfil.html', usuario={'nome': user.nome}, email=user.email, is_admin=is_admin)
    else:
        flash('Usuário não encontrado.')
        return redirect(url_for('logout'))

@app.route('/alterar_senha', methods=['GET', 'POST'])
def alterar_senha():
    if 'usuario_logado' not in session:
        flash('Faça login para alterar a senha.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        senha_atual = request.form.get('senha_atual')
        nova_senha = request.form.get('nova_senha')
        confirmar_senha = request.form.get('confirmar_senha')
        
        user = User.query.filter_by(email=session['usuario_logado']).first()
        if not user or not check_password_hash(user.senha, senha_atual):
            flash('Senha atual incorreta.', 'danger')
            return redirect(url_for('alterar_senha'))

        if nova_senha != confirmar_senha:
            flash('Nova senha e confirmação não conferem.', 'danger')
            return redirect(url_for('alterar_senha'))
        
        if not nova_senha:
            flash('A nova senha não pode ser vazia.', 'danger')
            return redirect(url_for('alterar_senha'))

        user.senha = generate_password_hash(nova_senha, method='pbkdf2:sha256')
        db.session.commit()
        
        flash('Senha alterada com sucesso.', 'success')
        return redirect(url_for('perfil'))

    is_admin = session.get('is_admin', False)
    return render_template('alterar_senha.html', is_admin=is_admin)

@app.route('/funcao_transferencia')
def funcao_transferencia():
    ft_latex = session.get('ft_latex', "Função de Transferência não disponível.")
    is_admin = session.get('is_admin', False)
    return render_template('transferencia.html', ft_latex=ft_latex, is_admin=is_admin)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=True)
