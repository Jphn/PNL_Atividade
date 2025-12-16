# Programação Não Linear (PNL) — modelagem, análise e implementação

**Curso:** Ciência da Computação — Pesquisa Operacional  
**Data:** 16/12/2025  

## 1. Introdução

Programação Não Linear (PNL) estuda problemas de otimização em que a função objetivo e/ou as restrições são não lineares. Diferentemente da Programação Linear, em PNL é comum encontrarmos múltiplos mínimos locais, regiões viáveis com fronteiras curvas e efeitos fortes de escala/condicionamento numérico. Isso torna essencial (i) classificar bem o problema (diferenciabilidade, convexidade, restrições) e (ii) escolher métodos adequados, distinguindo entre algoritmos **locais** (procuram um mínimo local) e **globais** (procuram reduzir o risco de ficar preso em mínimos locais).

Este trabalho cobre: (a) uma revisão teórica com classificação de problemas e condições KKT; (b) um protótipo 2D para visualizar não convexidade e trajetórias; (c) um estudo de caso de ajuste de curva logística via mínimos quadrados não linear, resolvido por métodos locais e globais.

---

## 2. Classificação de problemas de otimização

### 2.1 Forma geral

Uma PNL típica pode ser escrita como:

\[
\min_{x \in \mathbb{R}^n} \; f(x) \quad \text{sujeito a}\quad g_i(x) \le 0,\; i=1,\dots,m,\quad h_j(x)=0,\; j=1,\dots,p.
\]

- $f(x)$: função objetivo (minimizar custo, erro, energia, etc.).
- $g_i(x)$: restrições de desigualdade.
- $h_j(x)$: restrições de igualdade.

### 2.2 Diferenciabilidade

- **Diferenciável (C¹):** $f, g_i, h_j$ possuem gradientes bem definidos. Permite métodos baseados em gradiente, Newton, BFGS e KKT clássicas.
- **Duas vezes diferenciável (C²):** Hessianas existem (ou podem ser aproximadas). Facilita Newton e análises de convexidade via Hessiana.
- **Não diferenciável:** exige ferramentas de subgradiente, métodos proximais e outras generalizações (não é o foco aqui).

### 2.3 Convexidade

- Um problema é **convexo** se $f$ é convexa e o conjunto viável é convexo (tipicamente: todas as $g_i$ convexas e $h_j$ afins). Nesse caso, qualquer mínimo local é global.
- Um problema é **não convexo** se $f$ ou alguma restrição define um conjunto viável não convexo. Aqui podem existir vários mínimos locais; o palpite inicial torna-se decisivo para métodos locais.

Um teste prático (quando $f$ é C²) é verificar a Hessiana: $\nabla^2 f(x) \succeq 0$ em toda região de interesse implica convexidade.

### 2.4 Restrições

- **Sem restrições:** apenas $\min f(x)$.
- **Com desigualdades:** $g(x)\le 0$ gera fronteiras e ativações (restrições ativas no ótimo).
- **Com igualdades:** $h(x)=0$ reduz dimensionalidade e pode aumentar rigidez numérica.
- **Com limites (bounds):** $l \le x \le u$ são muito comuns e bem suportados em solvers.

---

## 3. Condições KKT (Karush–Kuhn–Tucker)

As condições KKT generalizam as condições de otimalidade de Lagrange para desigualdades. Para o problema

\[
\min f(x) \;\text{s.a.}\; g_i(x)\le 0,\; h_j(x)=0,
\]

definimos a Lagrangiana:

\[
\mathcal{L}(x,\lambda,\nu)=f(x)+\sum_{i=1}^m \lambda_i g_i(x)+\sum_{j=1}^p \nu_j h_j(x).
\]

Sob hipóteses de regularidade (ex.: LICQ), um ótimo local $x^\*$ satisfaz:

1. **Viabilidade primal:** $g_i(x^\*)\le 0$, $h_j(x^\*)=0$.
2. **Viabilidade dual:** $\lambda_i \ge 0$.
3. **Complementaridade:** $\lambda_i\, g_i(x^\*)=0$ (restrição ou ativa com multiplicador positivo, ou inativa com multiplicador zero).
4. **Estacionariedade:**
\[
\nabla f(x^\*) + \sum_i \lambda_i \nabla g_i(x^\*) + \sum_j \nu_j \nabla h_j(x^\*)=0.
\]

Em problemas convexos (com regularidade), KKT é **necessária e suficiente** para otimalidade global. Em problemas não convexos, KKT é tipicamente apenas necessária (pode haver pontos KKT que não são mínimos globais).

---

## 4. Métodos de solução: locais vs globais

### 4.1 Métodos locais

**(a) Gradiente (steepest descent)**

- Atualização: $x_{k+1}=x_k-\alpha_k \nabla f(x_k)$.
- Com **linha de busca** (ex.: Armijo/Wolfe) escolhe-se $\alpha_k$ para garantir redução suficiente.
- Prós: simples, robusto em convexos suaves.
- Contras: pode ser lento (zig-zag) e sensível ao condicionamento; em não convexos pode parar em mínimos locais/sela.

**(b) Newton / Newton-CG / trust-region**

- Usa Hessiana: resolve (aproximadamente) $\nabla^2 f(x_k) p_k = -\nabla f(x_k)$.
- Prós: convergência rápida perto do ótimo em funções bem comportadas.
- Contras: custo de Hessiana; Hessiana indefinida em não convexos pode apontar direção “errada”; exige salvaguardas (trust-region, regularização).

**(c) Quasi-Newton (BFGS/L-BFGS-B)**

- Aproxima Hessiana (ou sua inversa) com atualizações rank-2.
- Prós: geralmente bom custo/benefício; muito usado na prática.
- Contras: ainda é método local; depende de escala e inicialização.

Para **problemas com restrições**, métodos populares em `scipy.optimize` incluem:

- `SLSQP`: SQP com restrições gerais e bounds.
- `trust-constr`: trust-region com suporte a constraints via `NonlinearConstraint`/`LinearConstraint`.

### 4.2 Métodos globais

**(a) Simulated Annealing (SA)**

- Inspiração termodinâmica: aceita piores movimentos com probabilidade controlada por “temperatura” que decresce.
- Prós: simples e pode escapar de mínimos locais.
- Contras: convergência pode ser lenta; depende de agenda de resfriamento.

**(b) Differential Evolution (DE)**

- Algoritmo evolutivo baseado em recombinação/mutação de população.
- Prós: bom para não convexos e multimodais, fácil de usar, paralelizável.
- Contras: mais avaliações de função; lidar com restrições exige penalidade, reparo ou técnicas específicas.

Em geral, uma prática eficiente é **hibridizar**: usar DE para explorar globalmente e, ao final, refinar com um método local (BFGS/SLSQP/trust-constr).

---

## 5. Protótipo curto (2D): não convexidade + restrições

### 5.1 Modelo

Escolhemos um problema 2D com conjunto viável convexo, mas objetivo **não convexo**:

\[
\min_{x\in\mathbb{R}^2} \; f(x)= (x_1-1)^2 + (x_2-2)^2 + 0.5\,\sin(3x_1)\sin(3x_2)
\]

Sujeito a:

\[
\text{(C1)}\; x_1^2 + x_2^2 \le 5, \qquad \text{(C2)}\; x_1 + x_2 \ge 1.
\]

- O conjunto viável é a interseção de um disco (convexo) com um semiespaço (convexo), portanto é convexo.
- A função objetivo é **não convexa** pelo termo senoidal (Hessiana pode ser indefinida dependendo de $x$).

### 5.2 Resultados esperados

- Métodos locais (SLSQP/trust-constr) podem convergir para soluções diferentes dependendo do chute inicial.
- `differential_evolution` (com penalidade por violação) tende a achar soluções melhores/globalmente competitivas, e um refinamento local melhora precisão.
- Visualizações de contorno + trajetórias permitem observar a influência da não convexidade e das restrições ativas.

---

## 6. Estudo de caso: ajuste de curva logística (mínimos quadrados não linear)

### 6.1 Contexto

Ajustar parâmetros de um modelo não linear a dados (curva de crescimento, difusão de tecnologia, população, epidemias) é um caso clássico. Um modelo comum é a logística:

\[
\hat y(t;\theta) = \frac{L}{1+\exp(-k(t-t_0))}
\]

com parâmetros $\theta=(L,k,t_0)$:

- $L>0$: assíntota (capacidade).
- $k>0$: taxa.
- $t_0$: ponto de inflexão.

### 6.2 Formulação como PNL

Dado um conjunto de observações $(t_i, y_i)$, definimos o problema de mínimos quadrados não linear:

\[
\min_{L,k,t_0}\; \sum_{i=1}^N (y_i - \hat y(t_i;L,k,t_0))^2
\]

com bounds para refletir conhecimento físico:

\[
L \in [0, L_{max}],\quad k\in[0,k_{max}],\quad t_0\in[t_{min}, t_{max}].
\]

Essa função é suave, porém **não convexa** nos parâmetros (especialmente em $t_0$), então BFGS/L-BFGS-B podem ser sensíveis ao palpite inicial; DE é uma alternativa global.

### 6.3 Resultados esperados

- BFGS/L-BFGS-B: rápido, mas pode parar em mínimos locais.
- DE: mais custo de avaliação, mas mais robusto.
- Sensibilidade: rodar múltiplos chutes iniciais e comparar o valor objetivo e parâmetros estimados.

---

## 7. Conclusões

- A classificação (diferenciabilidade/convexidade/restrições) orienta a escolha do método.
- Em problemas não convexos, métodos locais são eficientes, mas dependem de chute inicial; métodos globais aumentam robustez.
- A combinação **global → local** é uma estratégia prática e didática.

Os resultados numéricos, gráficos de contorno e trajetórias dos iterados, e a análise de sensibilidade estão documentados e reproduzíveis no notebook e nos scripts.
