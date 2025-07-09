---
title: "Carla follow lane DDPG Vs PPO Vs SAC [July 1st]"
excerpt: "Refine PPO and compare in BM"

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

- Reentrenar PPO con entropy alta, std_dev baja y una red más grande para ver si así aprende algo parecido a DDPG y SAC
- Revisar por qué las métricas de BM marcan 1m de desviación media en town05 (no da esa sensación observando)
- Refinar gráficas de BM
- V media ← AÑADIR COLISIONES EN MISMA GRÁFICA
- V máxima ← AÑADIR
- Rizado (desviación típica)
- Histogramas v y desviación ← AÑADIR AUTOMÁTICA
- Invasión de carril
- Análisis en varias ciudades
- Subir fuente de las gráficas
- Usar town02 como entrenamiento y town01 para inferencia
- Corregir waypoints en mapas de ciudad02 en CARLA y documentarlo bien para poder recuperarlo para el artículo


More details in the [following slides](https://docs.google.com/presentation/d/1oWUP08XqX5Kit1zVz597sYLAW1FuFX-UyJQlLTNKOkY/edit?slide=id.g347779b8fc6_0_0#slide=id.g347779b8fc6_0_0)