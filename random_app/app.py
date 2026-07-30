# -*- coding: utf-8 -*-
"""
Otimizador de Políticas de Manutenção Preventiva Oportuna (Política QST)
-------------------------------------------------------------------------
Aplicativo Streamlit que reúne, em um único software, as duas versões
do modelo desenvolvido pelo grupo RANDOM:

  • QST (Base)      — sem a ocorrência de choques externos;
  • QST-Choques      — com a ocorrência de choques externos.

O usuário escolhe a política desejada pelo menu lateral.

@author: Thalia Queiroz.
"""

from pathlib import Path

import streamlit as st

# =============================================================================
# CAMINHOS DE ARQUIVOS 
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "logo_random.png"

# =============================================================================
# CONFIGURAÇÃO GERAL DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="RANDOM | Otimizador de Políticas QST",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# ESTILO (CSS) GLOBAL
# =============================================================================
def load_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            /* ---------- Tipografia geral ---------- */
            html, body, [class*="css"]  {
                font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            }
            h1, h2, h3, h4, h5, h6 {
                font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
                letter-spacing: -0.01em;
            }

            /* ---------- Esconde elementos padrão do Streamlit ---------- */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header[data-testid="stHeader"] {
                background: transparent;
            }
            header[data-testid="stHeader"] [data-testid="stToolbar"] {
                visibility: hidden;
            }
            
            /* ---------- FIXA O MENU LATERAL ---------- */
            /* Remove a seta de fechar o menu (dentro da sidebar) e o botão de abrir (fora) */
            [data-testid="stSidebarCollapseButton"],
            [data-testid="collapsedControl"] {
                display: none !important;
            }

            /* ---------- Container principal ---------- */
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 3rem;
                max-width: 1200px;
            }

            /* ---------- Cabeçalho de página ---------- */
            .page-header {
                position: relative;
                overflow: hidden;
                background: linear-gradient(135deg, #1F6F3F 0%, #14522D 100%);
                padding: 1.8rem 2.2rem;
                border-radius: 16px;
                margin-bottom: 1.9rem;
                box-shadow: 0 8px 24px rgba(20, 82, 45, 0.22);
            }
            .page-header::after {
                content: "";
                position: absolute;
                top: -60%;
                right: -8%;
                width: 260px;
                height: 260px;
                background: rgba(255,255,255,0.06);
                border-radius: 50%;
            }
            .page-header h1 {
                position: relative;
                color: #FFFFFF;
                font-size: 1.7rem;
                font-weight: 800;
                margin: 0 0 0.4rem 0;
                line-height: 1.3;
            }
            .page-header p {
                position: relative;
                color: #E3F2E8;
                font-size: 0.99rem;
                margin: 0;
            }
            .header-badge {
                display: inline-block;
                background: rgba(255,255,255,0.18);
                color: #FFFFFF;
                font-size: 0.72rem;
                font-weight: 600;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                padding: 0.2rem 0.6rem;
                border-radius: 999px;
                margin-bottom: 0.6rem;
            }

            /* ---------- Títulos de seção ---------- */
            .section-title {
                font-size: 1.15rem;
                font-weight: 700;
                color: #14522D;
                margin-top: 1.2rem;
                margin-bottom: 0.2rem;
                padding-bottom: 0.35rem;
                border-bottom: 2px solid #E3F2E8;
            }
            .section-help {
                font-size: 0.88rem;
                color: #5A6B63;
                margin-bottom: 0.8rem;
            }

            /* ---------- Cartões de métrica ---------- */
            .metric-card {
                background-color: #FFFFFF;
                border: 1px solid #E1E7E3;
                border-left: 4px solid #1F6F3F;
                border-radius: 12px;
                padding: 0.95rem 1.15rem;
                text-align: left;
                height: 100%;
                box-shadow: 0 2px 8px rgba(20, 82, 45, 0.06);
                transition: box-shadow 0.15s ease-in-out, transform 0.15s ease-in-out;
            }
            .metric-card:hover {
                box-shadow: 0 6px 16px rgba(20, 82, 45, 0.12);
                transform: translateY(-1px);
            }
            .metric-label {
                font-size: 0.8rem;
                color: #5A6B63;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.02em;
                margin-bottom: 0.25rem;
            }
            .metric-value {
                font-size: 1.55rem;
                font-weight: 700;
                color: #14522D;
            }
            .metric-help {
                font-size: 0.78rem;
                color: #7C8A83;
                margin-top: 0.2rem;
            }

            /* ---------- Botões ---------- */
            .stButton > button {
                background-color: #1F6F3F;
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 0.55rem 1.4rem;
                font-weight: 600;
                box-shadow: 0 2px 6px rgba(20, 82, 45, 0.18);
                transition: background-color 0.15s ease-in-out, transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
            }
            .stButton > button:hover {
                background-color: #14522D;
                color: #FFFFFF;
                transform: translateY(-1px);
                box-shadow: 0 5px 12px rgba(20, 82, 45, 0.25);
            }

            /* ---------- Sidebar ---------- */
            section[data-testid="stSidebar"] {
                background-color: #F8F9FA;
                border-right: 1px solid #E1E7E3;
            }
            /* O texto lateral agora é cor verde escura */
            section[data-testid="stSidebar"] * {
                color: #14522D !important;
            }
            section[data-testid="stSidebar"] .block-container {
                padding-top: 1.2rem;
            }
            .sidebar-title {
                font-size: 1.05rem;
                font-weight: 700;
                text-align: center;
                margin: 0.7rem 0 0.2rem 0;
                line-height: 1.35;
            }
            .sidebar-divider {
                height: 1px;
                background: linear-gradient(90deg, rgba(20,82,45,0) 0%, rgba(20,82,45,0.18) 50%, rgba(20,82,45,0) 100%);
                border: none;
                margin: 1.1rem 0;
            }
            .sidebar-nav-label {
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #5A7B68 !important;
                margin-bottom: 0.4rem;
            }
            /* Menu de navegação — transforma o radio padrão em uma lista de opções mais limpa */
            section[data-testid="stSidebar"] div[role="radiogroup"] {
                gap: 0.15rem;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] label {
                padding: 0.5rem 0.6rem;
                border-radius: 8px;
                transition: background-color 0.12s ease-in-out;
                width: 100%;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
                background-color: rgba(20, 82, 45, 0.07);
            }
            /* Ajuste do fundo da caixa de informação para um tom de verde bem leve e translúcido */
            .sidebar-info {
                background-color: rgba(20, 82, 45, 0.06);
                border: 1px solid rgba(20, 82, 45, 0.12);
                border-radius: 10px;
                padding: 0.8rem 0.9rem;
                font-size: 0.82rem;
                line-height: 1.45;
                color: #3E4A44 !important;
            }
            .sidebar-footer-caption {
                font-size: 0.72rem !important;
                color: #7C8A83 !important;
                line-height: 1.4;
            }

            /* ---------- Rodapé ---------- */
            .footer-divider {
                border: 0.5px solid #D8DEDB;
                margin-top: 2.2rem;
            }
            .footer {
                color: #8A948E;
                font-size: 0.78rem;
                text-align: left;
                line-height: 1.5;
                padding-bottom: 0.5rem;
            }
            .footer a {
                color: #1F6F3F;
                text-decoration: none;
            }
            .footer a:hover {
                text-decoration: underline;
            }

            /* ---------- Tabs ---------- */
            .stTabs [data-baseweb="tab"] {
                font-weight: 600;
            }

            /* ---------- Cartões da página inicial ---------- */
            .feature-card {
                background-color: #FFFFFF;
                border: 1px solid #E1E7E3;
                border-top: 4px solid #1F6F3F;
                border-radius: 12px;
                padding: 1.2rem 1.3rem;
                height: 100%;
                box-shadow: 0 2px 10px rgba(20, 82, 45, 0.07);
                transition: box-shadow 0.15s ease-in-out, transform 0.15s ease-in-out;
            }
            .feature-card:hover {
                box-shadow: 0 8px 20px rgba(20, 82, 45, 0.14);
                transform: translateY(-2px);
            }
            .feature-card-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #14522D;
                margin-bottom: 0.5rem;
            }
            .feature-card-body {
                font-size: 0.92rem;
                color: #3E4A44;
                line-height: 1.5;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

load_css()

# =============================================================================
# MENU LATERAL — SELEÇÃO DA POLÍTICA
# =============================================================================
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)

    st.markdown(
        "<div class='sidebar-title'>Otimizador de Políticas<br>de Manutenção</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='sidebar-divider'/>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-nav-label'>Navegação</div>", unsafe_allow_html=True)
    politica = st.radio(
        label="Selecione a página",
        options=[
            "Página Inicial",
            "Política QST (sem choques)",
            "Política QST (com choques)",
        ],
        label_visibility="collapsed",
        key="politica_selecionada",
    )

    st.markdown("<hr class='sidebar-divider'/>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='sidebar-info'>
        <b>Sobre a ferramenta</b><br>
        Otimização de políticas de manutenção preventiva oportuna
        em três fases (Q, S, T), fundamentadas na teoria de
        delay-time, com ou sem a modelagem de choques externos
        ao sistema.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='sidebar-divider'/>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sidebar-footer-caption'>RANDOM — Grupo de Pesquisa em Risco e "
        "Análise de Decisão em Operações e Manutenção</p>",
        unsafe_allow_html=True,
    )

# =============================================================================
# ROTEAMENTO PARA A PÁGINA SELECIONADA
# =============================================================================
if politica == "Página Inicial":
    from policies.home import render as render_home
    render_home()
elif politica == "Política QST (sem choques)":
    from policies.qst_sem_choques import render as render_sem_choques
    render_sem_choques()
elif politica == "Política QST (com choques)":
    from policies.qst_com_choques import render as render_com_choques
    render_com_choques()
