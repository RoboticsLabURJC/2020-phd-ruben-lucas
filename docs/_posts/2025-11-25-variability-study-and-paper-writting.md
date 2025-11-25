---
title: "variability study and paper writting (November week 2 - week 5)"
excerpt: "finished study and paper draft"

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

- Revisar si AUC es standard
- Variabilidad del entrenamiento
  - Entrenamiento
    - Algoritmos (3 entrenamientos * 4 semillas)
      - TD3
      - SAC
      - PPO
      - DDPG
      - SAC A Vs SAC C (optional)
      - SAC 3 layers Vs SAC 5 layers (optional)
    - Métricas para comparar los finalistas
      - Avg Cum reward over all 12 trainings on each algorithm
      - AUC con “step - cum reward” Revisar si es standard en el sector. Revisar más papers recientes con benchmarks
      - Times-to-converge not representative
      - It capturates sample-efficiency
      - It penalizes instability
      - matizar que para DDPG y TD3 se usa annealing pero para PPO y SAC el propio algoritmo calcula la exploración a aplicar
  - Inferencia
    - Utilizar una métrica en inferencia para representar distribución con N entrenamientos idénticos
    - Lane invasions
    - Desviación típica
    - Speed avg
  - Análisis sobre las métricas
    - Boxplots (con IC95?)
Terminar redacción del Paper.

More details in the [following slides](https://docs.google.com/presentation/d/1Ake2zFZjKYkIpSpIQafLuJgwgVv9ODgY1VLUqxVHfTg/edit?slide=id.g37fc31b9e03_0_0#slide=id.g37fc31b9e03_0_0)
