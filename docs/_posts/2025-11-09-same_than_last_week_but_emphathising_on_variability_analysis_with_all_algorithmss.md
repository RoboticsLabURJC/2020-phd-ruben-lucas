---
title: "Deep dive on variability analysis for all algorithms (November week 2)"
excerpt: "Deep dive on variability analysis for all algorithms"

sidebar:
  nav: "docs"

toc: true
toc_label: "TOC installation"
toc_icon: "cog"


categories:
- your category
tags:
- tag 1
- tag 2
- tag 3
- tag 4

author: Tony Stark
pinned: false
---

### Expected Goals

- Variabilidad del entrenamiento
  - Entrenamiento
    - Sacar mejor representante por algoritmo para evaluar en inferencia. El estudio de variabilidad va a garantizarnos un representante “fiable” de cada uno (revisa en otros papers número de semillas y si evaluan de otra manera)
  - Algoritmos (3 entrenamientos * 4 semillas) (de momento 3 * 3 semillas. Añadimos 4º después)
    - TD3
    - SAC
    - PPO
    - DDPG
    - SAC A Vs SAC C (optional)
    - SAC 3 layers Vs SAC 5 layers (optional)
  - Métricas
    - Cum reward + AUC
  - Inferencia
    - Utilizar una métrica en inferencia para representar distribución con N entrenamientos idénticos
    - Lane invasions
    - Desviación típica
    - Speed
  - Análisis sobre las métricas
    - Boxplots (con IC95?)
    - U de Mann-Whitney o H de Kruskal-Wallis
- Terminar redacción del Paper.

More details in the [following slides](https://docs.google.com/presentation/d/1ILu3VWtgL5L-kOO5tRcFMod3iTOCCFXq1_nCsMa7FAM/edit?slide=id.g37fc31b9e03_0_0#slide=id.g37fc31b9e03_0_0)