# -*- coding: utf-8 -*-
"""
Política QST (Base) — Manutenção Preventiva Oportuna em Três Fases,
sem a ocorrência de choques externos.

Modelo matemático original: Thalia Queiroz.
Este módulo mantém integralmente a formulação probabilística original,
reorganizando-a em funções parametrizadas por um dicionário de parâmetros
(`p`), o que facilita reuso na otimização, avaliação manual e análise
de sensibilidade.
"""

import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import quad, dblquad
from scipy.optimize import differential_evolution

from policies.common import render_header, render_footer, section_title, metric_card

# =============================================================================
# DISTRIBUIÇÕES DE PROBABILIDADE
# =============================================================================
def fx(t, p):
    return ((p["betax"] / p["etax"]) * ((t / p["etax"]) ** (p["betax"] - 1))) * np.exp(
        -((t / p["etax"]) ** p["betax"])
    )


def Rx(t, p):
    return np.exp(-((t / p["etax"]) ** p["betax"]))


def fh(t, p):
    return ((p["betah"] / p["etah"]) * ((t / p["etah"]) ** (p["betah"] - 1))) * np.exp(
        -((t / p["etah"]) ** p["betah"])
    )


def Rh(t, p):
    return np.exp(-((t / p["etah"]) ** p["betah"]))


def fw(t, p):
    return p["lambd"] * np.exp(-p["lambd"] * t)


def Rw(t, p):
    return np.exp(-p["lambd"] * t)


# =============================================================================
# CENÁRIOS DO MODELO (probabilidade, custo esperado, duração esperada)
# =============================================================================
def P1(Q, S, T, p):
    return Rx(T, p) * Rw(T - S, p)


def EC1(Q, S, T, p):
    return P1(Q, S, T, p) * (p["Cp"] + p["lambd"] * (S - Q) * p["Ci"])


def EL1(Q, S, T, p):
    return P1(Q, S, T, p) * (T + p["Dp"])


def P2(Q, S, T, p):
    integral, _ = quad(lambda x: fx(x, p) * Rh(T - x, p) * Rw(T - S, p), S, T)
    return integral


def EC2(Q, S, T, p):
    return P2(Q, S, T, p) * (p["Cp"] + p["lambd"] * (S - Q) * p["Ci"])


def EL2(Q, S, T, p):
    return P2(Q, S, T, p) * (T + p["Dp"])


def P3(Q, S, T, p):
    integral, _ = quad(lambda x: fx(x, p) * Rh(T - x, p) * Rw(T - x, p), Q, S)
    return integral


def EC3(Q, S, T, p):
    integral, _ = quad(
        lambda x: (p["Cp"] + p["lambd"] * (x - Q) * p["Ci"]) * fx(x, p) * Rh(T - x, p) * Rw(T - x, p),
        Q, S,
    )
    return integral


def EL3(Q, S, T, p):
    return P3(Q, S, T, p) * (T + p["Dp"])


def P4(Q, S, T, p):
    integral, _ = quad(lambda x: fx(x, p) * Rh(T - x, p) * Rw(T - Q, p), 0, Q)
    return integral


def EC4(Q, S, T, p):
    return P4(Q, S, T, p) * p["Cp"]


def EL4(Q, S, T, p):
    return P4(Q, S, T, p) * (T + p["Dp"])


def P5(Q, S, T, p):
    integral, _ = quad(lambda w: fw(w, p) * Rx(S + w, p), 0, T - S)
    return integral


def EC5(Q, S, T, p):
    return P5(Q, S, T, p) * (p["Co"] + p["lambd"] * (S - Q) * p["Ci"])


def EL5(Q, S, T, p):
    integral, _ = quad(lambda w: fw(w, p) * Rx(S + w, p) * (S + w + p["Dp"]), 0, T - S)
    return integral


def P6(Q, S, T, p):
    integral, _ = dblquad(
        lambda x, w: fw(w, p) * fx(x, p) * Rh(S + w - x, p),
        0, T - S, lambda w: S, lambda w: S + w,
    )
    return integral


def EC6(Q, S, T, p):
    return P6(Q, S, T, p) * (p["Co"] + p["lambd"] * (S - Q) * p["Ci"])


def EL6(Q, S, T, p):
    integral, _ = dblquad(
        lambda x, w: (S + w + p["Dp"]) * fw(w, p) * fx(x, p) * Rh(S + w - x, p),
        0, T - S, lambda w: S, lambda w: S + w,
    )
    return integral


def P7(Q, S, T, p):
    integral, _ = dblquad(
        lambda w, x: fx(x, p) * fw(w, p) * Rh(w, p),
        Q, S, lambda x: 0, lambda x: T - x,
    )
    return integral


def EC7(Q, S, T, p):
    integral, _ = dblquad(
        lambda w, x: (p["Co"] + p["lambd"] * (x - Q) * p["Ci"]) * fx(x, p) * fw(w, p) * Rh(w, p),
        Q, S, lambda x: 0, lambda x: T - x,
    )
    return integral


def EL7(Q, S, T, p):
    integral, _ = dblquad(
        lambda w, x: (x + w + p["Dp"]) * fx(x, p) * fw(w, p) * Rh(w, p),
        Q, S, lambda x: 0, lambda x: T - x,
    )
    return integral


def P8(Q, S, T, p):
    integral, _ = dblquad(
        lambda w, x: fx(x, p) * fw(w, p) * Rh(Q + w - x, p),
        0, Q, lambda x: 0, lambda x: T - Q,
    )
    return integral


def EC8(Q, S, T, p):
    return P8(Q, S, T, p) * p["Co"]


def EL8(Q, S, T, p):
    integral, _ = dblquad(
        lambda w, x: (x + w + p["Dp"]) * fx(x, p) * fw(w, p) * Rh(Q + w - x, p),
        0, Q, lambda x: 0, lambda x: T - Q,
    )
    return integral


def P9(Q, S, T, p):
    integral, _ = dblquad(
        lambda h, x: fx(x, p) * fh(h, p) * Rw(x + h - S, p),
        S, T, lambda x: 0, lambda x: T - x,
    )
    return integral


def EC9(Q, S, T, p):
    return P9(Q, S, T, p) * (p["Cf"] + p["lambd"] * (S - Q) * p["Ci"])


def EL9(Q, S, T, p):
    integral, _ = dblquad(
        lambda h, x: (x + h + p["Df"]) * fx(x, p) * fh(h, p) * Rw(x + h - S, p),
        S, T, lambda x: 0, lambda x: T - x,
    )
    return integral


def P10(Q, S, T, p):
    integral, _ = dblquad(
        lambda h, x: fx(x, p) * fh(h, p) * Rw(h, p),
        Q, S, lambda x: 0, lambda x: T - x,
    )
    return integral


def EC10(Q, S, T, p):
    integral, _ = dblquad(
        lambda h, x: (p["Cf"] + p["lambd"] * (x - Q) * p["Ci"]) * fx(x, p) * fh(h, p) * Rw(h, p),
        Q, S, lambda x: 0, lambda x: T - x,
    )
    return integral


def EL10(Q, S, T, p):
    integral, _ = dblquad(
        lambda h, x: (x + h + p["Df"]) * fx(x, p) * fh(h, p) * Rw(h, p),
        Q, S, lambda x: 0, lambda x: T - x,
    )
    return integral


def P11(Q, S, T, p):
    integral, _ = dblquad(
        lambda h, x: fx(x, p) * fh(h, p) * Rw(x + h - Q, p),
        0, Q, lambda x: Q - x, lambda x: T - x,
    )
    return integral


def EC11(Q, S, T, p):
    return p["Cf"] * P11(Q, S, T, p)


def EL11(Q, S, T, p):
    integral, _ = dblquad(
        lambda h, x: (x + h + p["Df"]) * fx(x, p) * fh(h, p) * Rw(x + h - Q, p),
        0, Q, lambda x: Q - x, lambda x: T - x,
    )
    return integral


def P12(Q, S, T, p):
    integral, _ = dblquad(
        lambda h, x: fx(x, p) * fh(h, p),
        0, Q, lambda x: 0, lambda x: Q - x,
    )
    return integral


def EC12(Q, S, T, p):
    return p["Cf"] * P12(Q, S, T, p)


def EL12(Q, S, T, p):
    integral, _ = dblquad(
        lambda h, x: (x + h + p["Df"]) * fx(x, p) * fh(h, p),
        0, Q, lambda x: 0, lambda x: Q - x,
    )
    return integral


_SCENARIOS = list(range(1, 13))
_FAILURE_SCENARIOS = [9, 10, 11, 12]


def P_total(Q, S, T, p):
    return sum(globals()[f"P{i}"](Q, S, T, p) for i in _SCENARIOS)


def P_falha(Q, S, T, p):
    return sum(globals()[f"P{i}"](Q, S, T, p) for i in _FAILURE_SCENARIOS)


def EC_ciclo(Q, S, T, p):
    return sum(globals()[f"EC{i}"](Q, S, T, p) for i in _SCENARIOS)


def EL_ciclo(Q, S, T, p):
    return sum(globals()[f"EL{i}"](Q, S, T, p) for i in _SCENARIOS)


def taxa_custo(Q, S, T, p):
    return EC_ciclo(Q, S, T, p) / EL_ciclo(Q, S, T, p)


def MTBOF(Q, S, T, p):
    return EL_ciclo(Q, S, T, p) / P_falha(Q, S, T, p)


# =============================================================================
# ANÁLISE DE SENSIBILIDADE
# =============================================================================
def analise_sensibilidade(Q, S, T, parametros_base, n_simulacoes, variacoes_parametros, parametros_alvo):
    resultados = []
    for _ in range(n_simulacoes):
        p_sim = parametros_base.copy()
        for param in parametros_alvo:
            variacao = variacoes_parametros.get(param, 0.1)
            perturbacao = np.random.uniform(1 - variacao, 1 + variacao)
            p_sim[param] *= perturbacao

        custo = taxa_custo(Q, S, T, p_sim)
        mtbof = MTBOF(Q, S, T, p_sim)
        resultados.append({"Custo": custo, "MTBOF": mtbof})

    df_resultados = pd.DataFrame(resultados)
    estatisticas = df_resultados.agg(["mean", "std"]).T
    estatisticas.columns = ["Média", "Desvio-padrão"]
    return df_resultados, estatisticas


# =============================================================================
# INTERFACE (STREAMLIT)
# =============================================================================
def render():
    render_header(
        "Política de Manutenção Preventiva Oportuna em Três Fases",
        "Modelo QST — otimização de política sem a ocorrência de choques externos ao sistema.",
        badge="Política Base · QST",
    )

    tab_params, tab_manual, tab_sens = st.tabs(
        ["⚙️ Parâmetros & Otimização", "🧪 Avaliação Manual", "📉 Análise de Sensibilidade"]
    )

    # -------------------------------------------------------------------
    # ABA 1 — PARÂMETROS E OTIMIZAÇÃO
    # -------------------------------------------------------------------
    with tab_params:
        section_title("📥", "Parâmetros do Modelo", "Defina as distribuições e os custos do sistema.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Defeito natural (X) — Weibull**")
            betax = st.number_input("Parâmetro de forma (β)", format="%.7f", step=0.0000001, value=2.0, key="sc_betax")
            etax = st.number_input("Parâmetro de escala (η)", format="%.7f", step=0.0000001, value=20.0, key="sc_etax")
            st.markdown("**Oportunidades (W) — Exponencial**")
            lambd = st.number_input("Taxa de chegada de oportunidades (λ)", format="%.7f", step=0.0000001, value=0.5, key="sc_lambd")
            st.markdown("**Custos**")
            Cp = st.number_input("Custo de substituição preventiva programada (Cp)", format="%.7f", step=0.0000001, value=1.0, key="sc_Cp")
            Co = st.number_input("Custo de substituição preventiva em oportunidade (Co)", format="%.7f", step=0.0000001, value=0.7, key="sc_Co")
            Ci = st.number_input("Custo de inspeção (Ci)", format="%.7f", step=0.0000001, value=0.05, key="sc_Ci")

        with col2:
            st.markdown("**Delay-time (H) — Weibull**")
            betah = st.number_input("Parâmetro de forma (β)", format="%.7f", step=0.0000001, value=1.3, key="sc_betah")
            etah = st.number_input("Parâmetro de escala (η)", format="%.7f", step=0.0000001, value=5.0, key="sc_etah")
            st.markdown("**Custos (continuação)**")
            Cf = st.number_input("Custo de substituição corretiva (Cf)", format="%.7f", step=0.0000001, value=10.0, key="sc_Cf")
            st.markdown("**Tempos de parada**")
            Dp = st.number_input("Tempo de parada — preventiva programada (Dp)", format="%.7f", step=0.0000001, value=0.1, key="sc_Dp")
            Df = st.number_input("Tempo de parada — corretiva (Df)", format="%.7f", step=0.0000001, value=0.5, key="sc_Df")

        params = dict(betax=betax, etax=etax, betah=betah, etah=etah, lambd=lambd, Ci=Ci, Co=Co, Cp=Cp, Cf=Cf, Dp=Dp, Df=Df)
        st.session_state["sc_params"] = params

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("🚀", "Otimização da Política", "Busca automática dos valores ótimos de Q, S e T.")

        if "sc_opt" not in st.session_state:
            st.session_state["sc_opt"] = None

        if st.button("🚀 Otimizar política", key="sc_btn_otimizar"):
            def objetivo(x):
                pQ, pS, T_val = x
                S_val = T_val * pS
                Q_val = S_val * pQ
                return taxa_custo(Q_val, S_val, T_val, params)

            bounds = [(0.0, 1.0), (0.0, 1.0), (0.0001, etax + etah)]

            with st.spinner("⏳ Otimizando a política QST... aguarde."):
                resultado = differential_evolution(objetivo, bounds=bounds, popsize=10, maxiter=50, tol=0.01, polish=True)

            pQ, pS, T_opt = resultado.x
            S_opt = T_opt * pS
            Q_opt = S_opt * pQ
            taxa_ot = taxa_custo(Q_opt, S_opt, T_opt, params)
            mtbof_ot = MTBOF(Q_opt, S_opt, T_opt, params)

            st.session_state["sc_opt"] = dict(Q=Q_opt, S=S_opt, T=T_opt, taxa=taxa_ot, mtbof=mtbof_ot)

        if st.session_state["sc_opt"]:
            r = st.session_state["sc_opt"]
            st.success("Otimização concluída!")
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: metric_card("Q ótimo", f"{r['Q']:.3f}")
            with c2: metric_card("S ótimo", f"{r['S']:.3f}")
            with c3: metric_card("T ótimo", f"{r['T']:.3f}")
            with c4: metric_card("Taxa de custo", f"{r['taxa']:.4f}")
            with c5: metric_card("MTBOF", f"{r['mtbof']:.2f}")

    # -------------------------------------------------------------------
    # ABA 2 — AVALIAÇÃO MANUAL
    # -------------------------------------------------------------------
    with tab_manual:
        section_title("🧪", "Avaliação de Política Pré-Definida", "Informe valores de Q, S e T e avalie o desempenho.")

        params = st.session_state.get("sc_params")
        if not params:
            st.info("Defina os parâmetros na aba **Parâmetros & Otimização** antes de avaliar uma política manual.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                Q_manual = st.number_input("Q (início de inspeções oportunas)", format="%.7f", step=0.0000001, key="sc_Q_manual")
            with c2:
                S_manual = st.number_input("S (limite para inspeções oportunas)", format="%.7f", step=0.0000001, key="sc_S_manual")
            with c3:
                T_manual = st.number_input("T (substituição programada)", format="%.7f", step=0.0000001, key="sc_T_manual")

            if "sc_manual" not in st.session_state:
                st.session_state["sc_manual"] = None

            if st.button("📊 Avaliar política", key="sc_btn_avaliar"):
                with st.spinner("🔍 Calculando desempenho da política..."):
                    taxa_manual = taxa_custo(Q_manual, S_manual, T_manual, params)
                    mtbof_manual = MTBOF(Q_manual, S_manual, T_manual, params)
                st.session_state["sc_manual"] = dict(Q=Q_manual, S=S_manual, T=T_manual, taxa=taxa_manual, mtbof=mtbof_manual)

            if st.session_state["sc_manual"]:
                r = st.session_state["sc_manual"]
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1: metric_card("Taxa de Custo", f"{r['taxa']:.4f}")
                with c2: metric_card("MTBOF", f"{r['mtbof']:.2f}")

    # -------------------------------------------------------------------
    # ABA 3 — ANÁLISE DE SENSIBILIDADE
    # -------------------------------------------------------------------
    with tab_sens:
        section_title("📉", "Análise de Sensibilidade", "Avalie o impacto da imprecisão dos parâmetros sobre a política pré-definida.")

        params = st.session_state.get("sc_params")
        manual = st.session_state.get("sc_manual")

        if not params or not manual:
            st.info("Realize a **Avaliação Manual** de uma política antes de executar a análise de sensibilidade.")
        else:
            n_simulacoes = st.number_input("Tamanho da amostra", min_value=100, max_value=500, value=100, step=100, key="sc_n_sim")

            st.markdown("**Parâmetros com imprecisão na estimativa (%)**")
            parametros_disponiveis = ["betax", "etax", "betah", "etah", "lambd", "Ci", "Co", "Cp", "Cf", "Dp", "Df"]
            variacoes_parametros = {}

            for param in parametros_disponiveis:
                c1, c2 = st.columns([2, 1])
                with c1:
                    incluir = st.checkbox(param, value=True, key=f"sc_chk_{param}")
                with c2:
                    variacao = st.slider(f"Imprecisão — {param}", 1, 100, 10, 1, key=f"sc_sld_{param}") / 100 if incluir else 0
                    variacoes_parametros[param] = variacao

            if st.button("🚀 Iniciar análise de sensibilidade", key="sc_btn_sens"):
                with st.spinner("⏳ Executando a análise de sensibilidade..."):
                    parametros_base = params.copy()
                    downtime_minimo = 1e-6
                    parametros_base["Dp"] = max(parametros_base["Dp"], downtime_minimo)
                    parametros_base["Df"] = max(parametros_base["Df"], downtime_minimo)

                    df_resultados, estatisticas = analise_sensibilidade(
                        manual["Q"], manual["S"], manual["T"],
                        parametros_base, int(n_simulacoes),
                        variacoes_parametros, parametros_disponiveis,
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                section_title("📊", "Resultados da Simulação")

                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                ax[0].boxplot(df_resultados["Custo"], vert=False, patch_artist=True,
                              boxprops=dict(facecolor="#8FC7A0"))
                ax[0].set_title("Taxa de Custo", loc="left", fontsize=12, color="#14522D", fontweight="bold")
                ax[0].text(0.01, 1.22,
                           f"Média = {df_resultados['Custo'].mean():.4f}\nDesvio-padrão = {df_resultados['Custo'].std():.4f}",
                           transform=ax[0].transAxes, fontsize=10, va="top", ha="left")

                ax[1].boxplot(df_resultados["MTBOF"], vert=False, patch_artist=True,
                              boxprops=dict(facecolor="#A8D5BA"))
                ax[1].set_title("MTBOF", loc="left", fontsize=12, color="#14522D", fontweight="bold")
                ax[1].text(0.01, 1.22,
                           f"Média = {df_resultados['MTBOF'].mean():.4f}\nDesvio-padrão = {df_resultados['MTBOF'].std():.4f}",
                           transform=ax[1].transAxes, fontsize=10, va="top", ha="left")

                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150)
                plt.close(fig)
                buf.seek(0)
                st.image(buf)

    render_footer()
