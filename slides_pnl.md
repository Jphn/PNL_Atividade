# Slides — Programação Não Linear (PNL)

## 1) Motivação

- PNL aparece em engenharia, economia, ML e ajuste de modelos
- Desafios: não convexidade, restrições, sensibilidade ao chute inicial

## 2) Classificação

- Diferenciabilidade: C¹/C² vs não diferenciável
- Convexidade: convexa vs não convexa
- Restrições: desigualdade, igualdade, bounds

## 3) KKT (ideia)

- Lagrangiana: objetivo + multiplicadores
- Viabilidade primal/dual, complementaridade, estacionariedade
- Convexo + regularidade: KKT também é suficiente

## 4) Métodos locais

- Gradiente + linha de busca (Armijo/Wolfe)
- BFGS/L-BFGS-B (quasi-Newton)
- Newton / trust-region
- Com restrições: SLSQP, trust-constr

## 5) Métodos globais

- Simulated annealing (conceito)
- Differential evolution (população + mutação)
- Estratégia prática: global → refino local

## 6) Protótipo 2D (demo)

- Objetivo não convexo + restrições
- Contornos + região viável
- Trajetórias SLSQP/trust-constr
- Comparar com DE + refino

## 7) Estudo de caso (ajuste logístico)

- Mínimos quadrados não linear
- Comparar local vs global
- Sensibilidade a palpites

## 8) Como rodar

- `pip install -r requirements.txt`
- Abrir notebook e executar
- `pytest -q`
