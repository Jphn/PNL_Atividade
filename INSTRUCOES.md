# Atividade — Programação Não Linear (PNL)

Este projeto contém:

- Um **relatório** (teoria + modelagem + resultados)
- Um **notebook** com implementação (métodos locais e globais) + visualizações
- **Código modular** em Python (funções/experimentos)
- **Testes** automatizados (pytest)
- **Slides** curtos para apresentação

## 1) Requisitos

- Python 3.10+ (recomendado 3.11)

## 2) Ambiente virtual e instalação

No Linux/macOS:

```bash
cd "PNL_Atividade"
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 3) Rodar o notebook (demo)

```bash
cd "PNL_Atividade"
jupyter notebook
```

Abra o arquivo [pnl_atividade.ipynb](pnl_atividade.ipynb) e execute as células em ordem.

## 4) Rodar testes

```bash
cd "PNL_Atividade"
pytest -q
```

## 5) Executar scripts (opcional)

Os experimentos também podem ser executados via scripts:

```bash
cd "PNL_Atividade"
python -m pnl.prototipo
python -m pnl.logistic_fit
```

## 6) Observações

- Os métodos **locais** (ex.: BFGS/SLSQP/trust-constr) podem ser **sensíveis ao palpite inicial** em problemas não convexos.
- O método **global** (`differential_evolution`) tende a ser mais robusto, porém mais custoso.
