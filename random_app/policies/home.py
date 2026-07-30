# -*- coding: utf-8 -*-
"""
Página inicial do aplicativo — apresentação do grupo de pesquisa RANDOM
e visão geral das duas políticas de manutenção disponíveis no software.
"""

import streamlit as st

from policies.common import render_header, render_footer, section_title


def render():
    render_header(
        "RANDOM — Grupo de Pesquisa em Risco e Análise de Decisão em Operações e Manutenção",
        "Bem-vindo ao Otimizador de Políticas de Manutenção Preventiva Oportuna (Política QST).",
        badge="Quem Somos",
    )

    # Passamos uma string vazia no primeiro argumento para não quebrar a função section_title
    section_title("", "Sobre o grupo")
    st.markdown(
        """
        O **RANDOM** foi criado em 2012 e reúne pesquisadores dedicados às áreas
        de risco, manutenção e modelagem de operações. O grupo desenvolve
        modelos matemáticos e ferramentas computacionais para apoiar a tomada
        de decisão em políticas de manutenção industrial.

        Saiba mais sobre o RANDOM no
        [diretório dos Grupos de Pesquisa do CNPq](http://dgp.cnpq.br/dgp/faces/consulta/consulta_parametrizada.jsf).
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)
    
    section_title("", "Sobre esta ferramenta")
    st.markdown(
        """
        Este software reúne, em um único lugar, dois modelos de otimização da
        **Política QST** — uma política de manutenção preventiva oportuna em
        três fases (definidas pelos instantes **Q**, **S** e **T**), baseada na
        teoria de *delay-time*. Use o menu à esquerda para escolher qual
        política deseja utilizar.
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-card-title">QST — Política sem Choques</div>
                <div class="feature-card-body">
                    Modelo sem a ocorrência de choques externos. Considera
                    apenas o defeito natural do sistema, o tempo
                    até a falha e as oportunidades de manutenção.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-card-title">QST — Política com Choques</div>
                <div class="feature-card-body">
                    Esse modelo considera a ocorrência de choques externos ao 
                    sistema, capazes de antecipar o defeito ou a falha.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    section_title("", "Como usar")
    st.markdown(
        """
        1. Escolha a política desejada no menu lateral (**QST — Política sem choques** ou **QST — Política com Choques**);
        2. Informe os parâmetros do modelo (distribuições, custos e tempos de parada);
        3. Otimize a política automaticamente ou avalie o desempenho de uma política definida manualmente;
        4. Explore a análise de sensibilidade para entender o impacto da imprecisão nos parâmetros sobre os resultados.
        """
    )

    render_footer()
