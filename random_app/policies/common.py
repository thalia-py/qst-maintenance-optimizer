# -*- coding: utf-8 -*-
"""
Componentes visuais compartilhados entre as páginas de política
(QST sem choques e QST-Choques), garantindo uma identidade visual
consistente em todo o aplicativo.
"""

import streamlit as st


def render_header(title: str, subtitle: str, badge: str = ""):
    """Cabeçalho de página com título, subtítulo e um selo (badge) opcional."""
    badge_html = f"<span class='header-badge'>{badge}</span>" if badge else ""
    st.markdown(
        f"""
        <div class="page-header">
            {badge_html}
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(icon: str, text: str, help_text: str = ""):
    """Título de seção padronizado, com ícone e (opcionalmente) uma descrição."""
    st.markdown(f"<div class='section-title'>{icon} {text}</div>", unsafe_allow_html=True)
    if help_text:
        st.markdown(f"<div class='section-help'>{help_text}</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, help_text: str = ""):
    """Renderiza um cartão de métrica com estilo customizado (substitui st.metric)."""
    help_html = f"<div class='metric-help'>{help_text}</div>" if help_text else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <hr class="footer-divider"/>
        <div class="footer">
            <strong>RANDOM</strong> — Grupo de Pesquisa em Risco e Análise de Decisão em Operações e Manutenção<br>
            Criado em 2012, o grupo reúne pesquisadores dedicados às áreas de risco, manutenção e modelagem de operações.<br>
            <a href="http://random.org.br" target="_blank">Acesse o site do RANDOM</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
