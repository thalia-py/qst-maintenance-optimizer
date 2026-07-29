# Otimizador de Políticas de Manutenção QST — RANDOM

Aplicativo Streamlit que reúne, em um único software com menu lateral,
as duas versões do modelo de manutenção preventiva oportuna em três
fases (política QST) desenvolvido pelo grupo RANDOM:

- **QST — Política Base**: sem a ocorrência de choques externos
  (baseado em `sft_qst.py`).
- **QST-Choques**: com a ocorrência de choques externos ao sistema
  (baseado em `qst_choques_otimizador.py`).

A lógica matemática de ambos os modelos foi mantida integralmente
(validada numericamente contra os scripts originais); o que mudou foi
a organização do código (parametrização por dicionário, um módulo por
política) e a interface, agora com um visual mais profissional e
navegação por abas.

## Estrutura de arquivos

```
random_app/
├── app.py                       # ponto de entrada (menu lateral + tema)
├── requirements.txt
├── .streamlit/
│   └── config.toml              # tema de cores do Streamlit
├── assets/
│   └── logo_random.png          # ⚠️ adicione aqui o logo do grupo
└── policies/
    ├── __init__.py
    ├── common.py                # cabeçalho, rodapé e cartões de métrica
    ├── qst_sem_choques.py       # política sem choques
    └── qst_com_choques.py       # política com choques
```

## Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Como hospedar no Streamlit Community Cloud

1. Suba esta pasta para um repositório no GitHub (mantendo a estrutura
   de pastas acima).
2. Adicione o arquivo de logo em `assets/logo_random.png` (se o
   arquivo não existir, o app simplesmente não exibe o logo — não
   quebra).
3. Em https://share.streamlit.io, crie um novo app apontando para o
   repositório e o arquivo `app.py`.
4. Pronto — o Streamlit Cloud instala as dependências de
   `requirements.txt` automaticamente.

## Observações

- Cada política guarda seus próprios parâmetros e resultados em
  `st.session_state` com prefixos distintos (`sc_` para "sem choques"
  e `cc_` para "com choques"), então alternar entre as duas no menu
  lateral não gera conflito de dados.
- A otimização usa `scipy.optimize.differential_evolution`, igual aos
  scripts originais; para modelos com muitos cenários (a versão com
  choques tem 22), a otimização pode levar de dezenas de segundos a
  poucos minutos, dependendo do hardware.
