# -*- coding: utf-8 -*-
"""
Política QST-Choques — Manutenção Preventiva Oportuna em Três Fases,
com a ocorrência de choques externos ao sistema.

Este módulo mantém integralmente a formulação probabilística original
(22 cenários), reorganizando-a em funções parametrizadas por um
dicionário de parâmetros (`p`), para reuso na otimização, avaliação
manual e análise de sensibilidade.
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
def fx(x, p):
    return ((p["betax"] / p["etax"]) * ((x / p["etax"]) ** (p["betax"] - 1))) * np.exp(
        -((x / p["etax"]) ** p["betax"])
    ) if x > 0 else 0.0


def Rx(x, p):
    return np.exp(-((x / p["etax"]) ** p["betax"])) if x >= 0 else 1.0


def fh(h, p):
    return ((p["betah"] / p["etah"]) * ((h / p["etah"]) ** (p["betah"] - 1))) * np.exp(
        -((h / p["etah"]) ** p["betah"])
    ) if h > 0 else 0.0


def Rh(h, p):
    return np.exp(-((h / p["etah"]) ** p["betah"])) if h >= 0 else 1.0


def fw(w, p):
    return p["lambd"] * np.exp(-p["lambd"] * w) if p["lambd"] > 0 and w >= 0 else 0.0


def Rw(w, p):
    return np.exp(-p["lambd"] * w) if w >= 0 else 1.0


def fc(c, p):
    return p["phi"] * np.exp(-p["phi"] * c) if p["phi"] > 0 and c >= 0 else 0.0


def Rc(c, p):
    return np.exp(-p["phi"] * c) if c >= 0 else 1.0


# =============================================================================
# CENÁRIOS DO MODELO (probabilidade, custo esperado, duração esperada)
# =============================================================================
# CENÁRIO 1
def P1(Q, S, T, p):
    return Rx(T, p) * Rc(T, p) * Rw(T - S, p)


def EC1(Q, S, T, p):
    return P1(Q, S, T, p) * (p["Cp"] + p["lambd"] * (S - Q) * p["Ci"])


def EL1(Q, S, T, p):
    return P1(Q, S, T, p) * (T + p["Dp"])


# CENÁRIO 2
def P2(Q, S, T, p):
    return quad(lambda x: fx(x, p) * Rc(x, p) * Rh(T - x, p) * Rw(T - S, p), S, T)[0]


def EC2(Q, S, T, p):
    return P2(Q, S, T, p) * (p["Cp"] + p["lambd"] * (S - Q) * p["Ci"])


def EL2(Q, S, T, p):
    return P2(Q, S, T, p) * (T + p["Dp"])


# CENÁRIO 3
def P3(Q, S, T, p):
    return quad(lambda c: fc(c, p) * Rx(c, p) * Rh(T - c, p) * Rw(T - S, p), S, T)[0]


def EC3(Q, S, T, p):
    return P3(Q, S, T, p) * (p["Cp"] + p["lambd"] * (S - Q) * p["Ci"])


def EL3(Q, S, T, p):
    return P3(Q, S, T, p) * (T + p["Dp"])


# CENÁRIO 4
def P4(Q, S, T, p):
    return quad(lambda x: fx(x, p) * Rc(x, p) * Rh(T - x, p) * Rw(T - x, p), Q, S)[0]


def EC4(Q, S, T, p):
    return quad(
        lambda x: (p["Cp"] + p["lambd"] * (x - Q) * p["Ci"]) * fx(x, p) * Rc(x, p) * Rh(T - x, p) * Rw(T - x, p),
        Q, S,
    )[0]


def EL4(Q, S, T, p):
    return P4(Q, S, T, p) * (T + p["Dp"])


# CENÁRIO 5
def P5(Q, S, T, p):
    return quad(lambda c: fc(c, p) * Rx(c, p) * Rh(T - c, p) * Rw(T - c, p), Q, S)[0]


def EC5(Q, S, T, p):
    return quad(
        lambda c: (p["Cp"] + p["lambd"] * (c - Q) * p["Ci"]) * fc(c, p) * Rx(c, p) * Rh(T - c, p) * Rw(T - c, p),
        Q, S,
    )[0]


def EL5(Q, S, T, p):
    return P5(Q, S, T, p) * (T + p["Dp"])


# CENÁRIO 6
def P6(Q, S, T, p):
    return quad(lambda x: fx(x, p) * Rc(x, p) * Rh(T - x, p) * Rw(T - Q, p), 0, Q)[0]


def EC6(Q, S, T, p):
    return P6(Q, S, T, p) * p["Cp"]


def EL6(Q, S, T, p):
    return P6(Q, S, T, p) * (T + p["Dp"])


# CENÁRIO 7
def P7(Q, S, T, p):
    return quad(lambda c: fc(c, p) * Rx(c, p) * Rh(T - c, p) * Rw(T - Q, p), 0, Q)[0]


def EC7(Q, S, T, p):
    return P7(Q, S, T, p) * p["Cp"]


def EL7(Q, S, T, p):
    return P7(Q, S, T, p) * (T + p["Dp"])


# CENÁRIO 8
def P8(Q, S, T, p):
    return quad(lambda w: fw(w, p) * Rx(S + w, p) * Rc(S + w, p), 0, T - S)[0]


def EC8(Q, S, T, p):
    return P8(Q, S, T, p) * (p["Co"] + p["lambd"] * (S - Q) * p["Ci"])


def EL8(Q, S, T, p):
    return quad(lambda w: (S + w + p["Dp"]) * fw(w, p) * Rx(S + w, p) * Rc(S + w, p), 0, T - S)[0]


# CENÁRIO 9
def P9(Q, S, T, p):
    return dblquad(
        lambda x, w: fw(w, p) * fx(x, p) * Rc(x, p) * Rh(S + w - x, p),
        0, T - S, lambda w: S, lambda w: S + w,
    )[0]


def EC9(Q, S, T, p):
    return P9(Q, S, T, p) * (p["Co"] + p["lambd"] * (S - Q) * p["Ci"])


def EL9(Q, S, T, p):
    return dblquad(
        lambda x, w: (S + w + p["Dp"]) * fw(w, p) * fx(x, p) * Rc(x, p) * Rh(S + w - x, p),
        0, T - S, lambda w: S, lambda w: S + w,
    )[0]


# CENÁRIO 10
def P10(Q, S, T, p):
    return dblquad(
        lambda c, w: fw(w, p) * fc(c, p) * Rx(c, p) * Rh(S + w - c, p),
        0, T - S, lambda w: S, lambda w: S + w,
    )[0]


def EC10(Q, S, T, p):
    return P10(Q, S, T, p) * (p["Co"] + p["lambd"] * (S - Q) * p["Ci"])


def EL10(Q, S, T, p):
    return dblquad(
        lambda c, w: (S + w + p["Dp"]) * fw(w, p) * fc(c, p) * Rx(c, p) * Rh(S + w - c, p),
        0, T - S, lambda w: S, lambda w: S + w,
    )[0]


# CENÁRIO 11
def P11(Q, S, T, p):
    return dblquad(
        lambda w, x: fx(x, p) * Rc(x, p) * fw(w, p) * Rh(w, p),
        Q, S, lambda x: 0, lambda x: T - x,
    )[0]


def EC11(Q, S, T, p):
    return dblquad(
        lambda w, x: (p["Co"] + p["lambd"] * (x - Q) * p["Ci"]) * fx(x, p) * Rc(x, p) * fw(w, p) * Rh(w, p),
        Q, S, lambda x: 0, lambda x: T - x,
    )[0]


def EL11(Q, S, T, p):
    return dblquad(
        lambda w, x: (x + w + p["Dp"]) * fx(x, p) * Rc(x, p) * fw(w, p) * Rh(w, p),
        Q, S, lambda x: 0, lambda x: T - x,
    )[0]


# CENÁRIO 12
def P12(Q, S, T, p):
    return dblquad(
        lambda w, c: fc(c, p) * Rx(c, p) * fw(w, p) * Rh(w, p),
        Q, S, lambda c: 0, lambda c: T - c,
    )[0]


def EC12(Q, S, T, p):
    return dblquad(
        lambda w, c: (p["Co"] + p["lambd"] * (c - Q) * p["Ci"]) * fc(c, p) * Rx(c, p) * fw(w, p) * Rh(w, p),
        Q, S, lambda c: 0, lambda c: T - c,
    )[0]


def EL12(Q, S, T, p):
    return dblquad(
        lambda w, c: (c + w + p["Dp"]) * fc(c, p) * Rx(c, p) * fw(w, p) * Rh(w, p),
        Q, S, lambda c: 0, lambda c: T - c,
    )[0]


# CENÁRIO 13
def P13(Q, S, T, p):
    return dblquad(
        lambda w, x: fx(x, p) * Rc(x, p) * fw(w, p) * Rh(Q + w - x, p),
        0, Q, lambda x: 0, lambda x: T - Q,
    )[0]


def EC13(Q, S, T, p):
    return P13(Q, S, T, p) * p["Co"]


def EL13(Q, S, T, p):
    return dblquad(
        lambda w, x: (Q + w + p["Dp"]) * fx(x, p) * Rc(x, p) * fw(w, p) * Rh(Q + w - x, p),
        0, Q, lambda x: 0, lambda x: T - Q,
    )[0]


# CENÁRIO 14
def P14(Q, S, T, p):
    return dblquad(
        lambda w, c: fc(c, p) * Rx(c, p) * fw(w, p) * Rh(Q + w - c, p),
        0, Q, lambda c: 0, lambda c: T - Q,
    )[0]


def EC14(Q, S, T, p):
    return P14(Q, S, T, p) * p["Co"]


def EL14(Q, S, T, p):
    return dblquad(
        lambda w, c: (Q + w + p["Dp"]) * fc(c, p) * Rx(c, p) * fw(w, p) * Rh(Q + w - c, p),
        0, Q, lambda c: 0, lambda c: T - Q,
    )[0]


# CENÁRIO 15
def P15(Q, S, T, p):
    return dblquad(
        lambda h, x: fx(x, p) * Rc(x, p) * fh(h, p) * Rw(x + h - S, p),
        S, T, lambda x: 0, lambda x: T - x,
    )[0]


def EC15(Q, S, T, p):
    return P15(Q, S, T, p) * (p["Cf"] + p["lambd"] * (S - Q) * p["Ci"])


def EL15(Q, S, T, p):
    return dblquad(
        lambda h, x: (x + h + p["Df"]) * fx(x, p) * Rc(x, p) * fh(h, p) * Rw(x + h - S, p),
        S, T, lambda x: 0, lambda x: T - x,
    )[0]


# CENÁRIO 16
def P16(Q, S, T, p):
    return dblquad(
        lambda h, c: fc(c, p) * Rx(c, p) * fh(h, p) * Rw(c + h - S, p),
        S, T, lambda c: 0, lambda c: T - c,
    )[0]


def EC16(Q, S, T, p):
    return P16(Q, S, T, p) * (p["Cf"] + p["lambd"] * (S - Q) * p["Ci"])


def EL16(Q, S, T, p):
    return dblquad(
        lambda h, c: (c + h + p["Df"]) * fc(c, p) * Rx(c, p) * fh(h, p) * Rw(c + h - S, p),
        S, T, lambda c: 0, lambda c: T - c,
    )[0]


# CENÁRIO 17
def P17(Q, S, T, p):
    return dblquad(
        lambda h, x: fx(x, p) * Rc(x, p) * fh(h, p) * Rw(h, p),
        Q, S, lambda x: 0, lambda x: T - x,
    )[0]


def EC17(Q, S, T, p):
    return dblquad(
        lambda h, x: (p["Cf"] + p["lambd"] * (x - Q) * p["Ci"]) * fx(x, p) * Rc(x, p) * fh(h, p) * Rw(h, p),
        Q, S, lambda x: 0, lambda x: T - x,
    )[0]


def EL17(Q, S, T, p):
    return dblquad(
        lambda h, x: (x + h + p["Df"]) * fx(x, p) * Rc(x, p) * fh(h, p) * Rw(h, p),
        Q, S, lambda x: 0, lambda x: T - x,
    )[0]


# CENÁRIO 18
def P18(Q, S, T, p):
    return dblquad(
        lambda h, c: fc(c, p) * Rx(c, p) * fh(h, p) * Rw(h, p),
        Q, S, lambda c: 0, lambda c: T - c,
    )[0]


def EC18(Q, S, T, p):
    return dblquad(
        lambda h, c: (p["Cf"] + p["lambd"] * (c - Q) * p["Ci"]) * fc(c, p) * Rx(c, p) * fh(h, p) * Rw(h, p),
        Q, S, lambda c: 0, lambda c: T - c,
    )[0]


def EL18(Q, S, T, p):
    return dblquad(
        lambda h, c: (c + h + p["Df"]) * fc(c, p) * Rx(c, p) * fh(h, p) * Rw(h, p),
        Q, S, lambda c: 0, lambda c: T - c,
    )[0]


# CENÁRIO 19
def P19(Q, S, T, p):
    return dblquad(
        lambda h, x: fx(x, p) * Rc(x, p) * fh(h, p) * Rw(x + h - Q, p),
        0, Q, lambda x: Q - x, lambda x: T - x,
    )[0]


def EC19(Q, S, T, p):
    return p["Cf"] * P19(Q, S, T, p)


def EL19(Q, S, T, p):
    return dblquad(
        lambda h, x: (x + h + p["Df"]) * fx(x, p) * Rc(x, p) * fh(h, p) * Rw(x + h - Q, p),
        0, Q, lambda x: Q - x, lambda x: T - x,
    )[0]


# CENÁRIO 20
def P20(Q, S, T, p):
    return dblquad(
        lambda h, c: fc(c, p) * Rx(c, p) * fh(h, p) * Rw(c + h - Q, p),
        0, Q, lambda c: Q - c, lambda c: T - c,
    )[0]


def EC20(Q, S, T, p):
    return p["Cf"] * P20(Q, S, T, p)


def EL20(Q, S, T, p):
    return dblquad(
        lambda h, c: (c + h + p["Df"]) * fc(c, p) * Rx(c, p) * fh(h, p) * Rw(c + h - Q, p),
        0, Q, lambda c: Q - c, lambda c: T - c,
    )[0]


# CENÁRIO 21
def P21(Q, S, T, p):
    return dblquad(
        lambda h, x: fx(x, p) * Rc(x, p) * fh(h, p),
        0, Q, lambda x: 0, lambda x: Q - x,
    )[0]


def EC21(Q, S, T, p):
    return p["Cf"] * P21(Q, S, T, p)


def EL21(Q, S, T, p):
    return dblquad(
        lambda h, x: (x + h + p["Df"]) * fx(x, p) * Rc(x, p) * fh(h, p),
        0, Q, lambda x: 0, lambda x: Q - x,
    )[0]


# CENÁRIO 22
def P22(Q, S, T, p):
    return dblquad(
        lambda h, c: fc(c, p) * Rx(c, p) * fh(h, p),
        0, Q, lambda c: 0, lambda c: Q - c,
    )[0]


def EC22(Q, S, T, p):
    return p["Cf"] * P22(Q, S, T, p)


def EL22(Q, S, T, p):
    return dblquad(
        lambda h, c: (c + h + p["Df"]) * fc(c, p) * Rx(c, p) * fh(h, p),
        0, Q, lambda c: 0, lambda c: Q - c,
    )[0]


_SCENARIOS = list(range(1, 23))
_FAILURE_SCENARIOS = list(range(15, 23))


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
        "Modelo QST-Choques — otimização de política considerando a ocorrência de choques externos ao sistema.",
        badge="Política com Choques · QST-Choques",
    )

    tab_params, tab_manual, tab_sens = st.tabs(
        ["⚙️ Parâmetros & Otimização", "🧪 Avaliação Manual", "📉 Análise de Sensibilidade"]
    )

    # -------------------------------------------------------------------
    # ABA 1 — PARÂMETROS E OTIMIZAÇÃO
    # -------------------------------------------------------------------
    with tab_params:
        section_title("📥", "Parâmetros do Modelo", "Defina as distribuições, os choques e os custos do sistema.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Defeito natural (X) — Weibull**")
            betax = st.number_input("Parâmetro de forma (β)", format="%.7f", step=0.0000001, value=2.0, key="cc_betax")
            etax = st.number_input("Parâmetro de escala (η)", format="%.7f", step=0.0000001, value=20.0, key="cc_etax")
            st.markdown("**Oportunidades (W) — Exponencial**")
            lambd = st.number_input("Taxa de chegada de oportunidades (λ)", format="%.7f", step=0.0000001, value=0.5, key="cc_lambd")
            st.markdown("**Choques (C) — Exponencial**")
            phi = st.number_input("Taxa de chegada de choques (φ)", format="%.7f", step=0.0000001, value=0.5, key="cc_phi")
            st.markdown("**Custos**")
            Cp = st.number_input("Custo de substituição preventiva programada (Cp)", format="%.7f", step=0.0000001, value=1.0, key="cc_Cp")

        with col2:
            st.markdown("**Delay-time (H) — Weibull**")
            betah = st.number_input("Parâmetro de forma (β)", format="%.7f", step=0.0000001, value=1.3, key="cc_betah")
            etah = st.number_input("Parâmetro de escala (η)", format="%.7f", step=0.0000001, value=5.0, key="cc_etah")
            st.markdown("**Custos (continuação)**")
            Co = st.number_input("Custo de substituição preventiva em oportunidade (Co)", format="%.7f", step=0.0000001, value=0.7, key="cc_Co")
            Ci = st.number_input("Custo de inspeção (Ci)", format="%.7f", step=0.0000001, value=0.05, key="cc_Ci")
            Cf = st.number_input("Custo de substituição corretiva (Cf)", format="%.7f", step=0.0000001, value=10.0, key="cc_Cf")
            st.markdown("**Tempos de parada**")
            cdp1, cdp2 = st.columns(2)
            with cdp1:
                Dp = st.number_input("Parada preventiva (Dp)", format="%.7f", step=0.0000001, value=0.1, key="cc_Dp")
            with cdp2:
                Df = st.number_input("Parada corretiva (Df)", format="%.7f", step=0.0000001, value=0.5, key="cc_Df")

        params = dict(betax=betax, etax=etax, betah=betah, etah=etah, lambd=lambd, phi=phi,
                      Ci=Ci, Co=Co, Cp=Cp, Cf=Cf, Dp=Dp, Df=Df)
        st.session_state["cc_params"] = params

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("🚀", "Otimização da Política", "Busca automática dos valores ótimos de Q, S e T.")

        if "cc_opt" not in st.session_state:
            st.session_state["cc_opt"] = None

        if st.button("🚀 Otimizar política", key="cc_btn_otimizar"):
            def objetivo(x):
                pQ, pS, T_val = x
                S_val = T_val * pS
                Q_val = S_val * pQ
                return taxa_custo(Q_val, S_val, T_val, params)

            Tmax = etax + etah
            bounds = [(0.0, 1.0), (0.0, 1.0), (0.0001, Tmax)]

            with st.spinner("⏳ Otimizando a política QST-Choques... aguarde."):
                resultado = differential_evolution(objetivo, bounds=bounds, popsize=10, maxiter=50, tol=0.01, polish=True)

            pQ, pS, T_opt = resultado.x
            S_opt = T_opt * pS
            Q_opt = S_opt * pQ
            taxa_ot = taxa_custo(Q_opt, S_opt, T_opt, params)
            mtbof_ot = MTBOF(Q_opt, S_opt, T_opt, params)

            st.session_state["cc_opt"] = dict(Q=Q_opt, S=S_opt, T=T_opt, taxa=taxa_ot, mtbof=mtbof_ot)

        if st.session_state["cc_opt"]:
            r = st.session_state["cc_opt"]
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

        params = st.session_state.get("cc_params")
        if not params:
            st.info("Defina os parâmetros na aba **Parâmetros & Otimização** antes de avaliar uma política manual.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                Q_manual = st.number_input("Q (início de inspeções oportunas)", format="%.7f", step=0.0000001, key="cc_Q_manual")
            with c2:
                S_manual = st.number_input("S (limite para inspeções oportunas)", format="%.7f", step=0.0000001, key="cc_S_manual")
            with c3:
                T_manual = st.number_input("T (substituição programada)", format="%.7f", step=0.0000001, key="cc_T_manual")

            if "cc_manual" not in st.session_state:
                st.session_state["cc_manual"] = None

            if st.button("📊 Avaliar política", key="cc_btn_avaliar"):
                with st.spinner("🔍 Calculando desempenho da política..."):
                    taxa_manual = taxa_custo(Q_manual, S_manual, T_manual, params)
                    mtbof_manual = MTBOF(Q_manual, S_manual, T_manual, params)
                st.session_state["cc_manual"] = dict(Q=Q_manual, S=S_manual, T=T_manual, taxa=taxa_manual, mtbof=mtbof_manual)

            if st.session_state["cc_manual"]:
                r = st.session_state["cc_manual"]
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1: metric_card("Taxa de Custo", f"{r['taxa']:.4f}")
                with c2: metric_card("MTBOF", f"{r['mtbof']:.2f}")

    # -------------------------------------------------------------------
    # ABA 3 — ANÁLISE DE SENSIBILIDADE
    # -------------------------------------------------------------------
    with tab_sens:
        section_title("📉", "Análise de Sensibilidade", "Avalie o impacto da imprecisão dos parâmetros sobre a política pré-definida.")

        params = st.session_state.get("cc_params")
        manual = st.session_state.get("cc_manual")

        if not params or not manual:
            st.info("Realize a **Avaliação Manual** de uma política antes de executar a análise de sensibilidade.")
        else:
            n_simulacoes = st.number_input("Tamanho da amostra", min_value=100, max_value=500, value=100, step=100, key="cc_n_sim")

            st.markdown("**Parâmetros com imprecisão na estimativa (%)**")
            parametros_disponiveis = ["betax", "etax", "betah", "etah", "lambd", "phi", "Ci", "Co", "Cp", "Cf", "Dp", "Df"]
            variacoes_parametros = {}

            for param in parametros_disponiveis:
                c1, c2 = st.columns([2, 1])
                with c1:
                    incluir = st.checkbox(param, value=True, key=f"cc_chk_{param}")
                with c2:
                    variacao = st.slider(f"Imprecisão — {param}", 1, 100, 10, 1, key=f"cc_sld_{param}") / 100 if incluir else 0
                    variacoes_parametros[param] = variacao

            if st.button("🚀 Iniciar análise de sensibilidade", key="cc_btn_sens"):
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
