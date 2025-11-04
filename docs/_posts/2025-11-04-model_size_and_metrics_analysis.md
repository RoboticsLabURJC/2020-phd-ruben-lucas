---
title: "Carla follow lane DDPG Vs PPO Vs SAC [October week 4 - November week 1]"
excerpt: "Once we got the optimal implementation, perform regression experiments"

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

- Tras leer un ártículo me surgió la duda de si con los últimos cambios, una red más pequeña podría hacer el trabajo
  - Con menos capas y más neuronas NO
  - Con menos capas y mismas neuronas SI
- Variabilidad del entrenamiento
  - Utilizar una métrica en inferencia para representar distribución (campana de gaus?) con N entrenamientos idénticos
- Algoritmos
  - TD3
  - SAC
  - PPO (optional)
  - DDPG (optional)
- Métricas
  - Lane invasions
  - Desviación típica
  - Speed
- Estudiar en la literatura
  - Cuantos entrenamientos son representativos?
  - Explorar si hay otras técnicas para elegir el mejor modelo durante el entrenamiento
  - La más utilizada es la misma que utilizamos nosotros
- Redacción del SOTA.

More details in the [following slides](https://docs.google.com/presentation/d/1Pf1QfedsZiB8eJP2-sIcOsSY8wiOgulUVP3qDSD9CGM/edit?slide=id.g37fc31b9e03_0_0#slide=id.g37fc31b9e03_0_0)