# 001 – Decisão: Não usar similaridade por cosseno como feature explícita

## Contexto
Durante a fase de feature engineering, foi avaliado o uso de `cosine_similarity(emb_vaga, emb_cv)` como feature para o modelo de rankeamento. Ou seja, a similaridade entre os embeddings da descrição da vaga versos os embeddings do curriculo do candidato.

## Decisão
Decidiu-se **não usar** a similaridade como uma feature explícita, pois:
- Os embeddings `emb_vaga` e `emb_cv` já serão  usados diretamente como features, porém com redução de dimensionalidade.
- O LightGBM pode aprender relações não lineares e interações vetoriais mais ricas

## Consequências
- O pipeline continua com as features `emb_*` explodidas em colunas.
- Reduzimos de 384 colunas para 75 (sujeito a alteração) utilizando PCA.
- A função de cálculo da similaridade foi mantida para o módulo de recomendação, mas não é input do modelo
