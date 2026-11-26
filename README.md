# Quanto o pneu "cai" por volta?

**Degradação de pneus na Fórmula 1 (GP de Barcelona 2026): abordagem bayesiana vs. frequentista**

Trabalho final da disciplina de Inferência Bayesiana. Com as voltas do GP de Barcelona 2026, o trabalho estima quantos segundos cada composto de pneu perde por volta, descontando o efeito do combustível, do piloto e do carro. O mesmo modelo é ajustado de duas formas e os resultados são comparados:

- **Frequentista:** máxima verossimilhança (EMV), com erros-padrão pela informação de Fisher observada.
- **Bayesiana:** priori conjugada Normal–Gama-Inversa, amostrador de Gibbs e distribuição preditiva a posteriori, em dois cenários de priori: **vaga** e **informativa** (power prior com a corrida de 2025, α₀ = 0,1).

## Modelo

Regressão linear normal com 38 coeficientes e a variância do ruído σ²:

- tempo de volta em função do composto (macio, médio, duro), da idade do pneu e da volta da corrida;
- efeito de ritmo de cada piloto e degradação extra de cada equipe.

Como a idade do pneu volta a zero no box e a contagem de voltas da corrida continua, o modelo consegue separar o desgaste do pneu do ganho de combustível e de pista.

## Principais resultados

| Grandeza (EMV) | Estimativa | IC 95% |
|---|---|---|
| Degradação do macio (s/volta) | 0,212 | [0,188; 0,236] |
| Degradação do médio (s/volta) | 0,179 | [0,165; 0,192] |
| Degradação do duro (s/volta) | 0,140 | [0,133; 0,148] |
| Ganho por volta de corrida (s/volta) | −0,060 | [−0,062; −0,058] |

- Com priori vaga, a abordagem bayesiana reproduz praticamente os números do EMV, com outra interpretação (intervalo HPD de credibilidade em vez de intervalo de confiança).
- O pneu médio perdeu cerca de 2,5 vezes mais tempo por volta do que em 2025. A priori informativa puxa as estimativas para os valores de 2025 e piora o WAIC e o LOO-CV.
- Gibbs com 4 cadeias (20 000 sorteios): R̂ ≤ 1,001 e ESS > 18 000; a posteriori exata e a obtida por MCMC coincidem.
- AIC, WAIC e LOO-CV escolhem o modelo com degradação por composto e por equipe; o BIC prefere o modelo sem o efeito de equipe.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `trabalho_final_f1_bayes.pdf` | Relatório final (análise e código comentado) |
| `trabalho_final_f1_bayes.ipynb` | Notebook executado que gera o relatório |
| `fonte/notebook.md` | Fonte do notebook (formato jupytext) |
| `scripts/construir_notebook.py` | Converte a fonte em `.ipynb` |
| `scripts/gerar_pdf.py` | Converte o notebook executado em PDF |

## Como reproduzir

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace trabalho_final_f1_bayes.ipynb
```

Na primeira execução, o notebook baixa os dados da [API pública OpenF1](https://openf1.org) e guarda um cache em `dados/openf1/`. A semente é fixa, então os resultados se repetem.

Para regenerar tudo a partir da fonte:

```bash
python scripts/construir_notebook.py fonte/notebook.md trabalho_final_f1_bayes.ipynb
jupyter nbconvert --to notebook --execute --inplace trabalho_final_f1_bayes.ipynb
python scripts/gerar_pdf.py     # usa o Microsoft Edge (Windows) para imprimir o PDF
```

## Dados

Tempos de volta e trocas de pneu do GP de Barcelona de 2025 e 2026, obtidos pela OpenF1, um projeto independente sem vínculo com a Fórmula 1. O notebook corrige uma inconsistência da própria fonte (uma troca de pneu registrada na volta errada) antes da análise.
