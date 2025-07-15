---
title: "Carla follow lane DDPG Vs PPO Vs SAC [July 2nd]"
excerpt: "Refine to excel town02 and bm metrics"

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

- Usar una ejecución impar de ejecuciones (12 → sentido aleatorio y puntos de inicio aleatorios)
- Media y desviación típica con boxplots. Aspiramos a gráfica en la que la desviación típica es baja
- En los bar plot, no poner una barra por experimento, sino una por ciudad
- Medir suavidad (revisar en Carla Leaderboard)
- Sacar las gráficas de tiempo de entrenamiento
- Sacar velocidad en el espacio (con offset para comparar fácilmente). Difícil hacerlo con inicios aleatorios a menos que lo orqueste  
  ¿Nos parece bien hacer un mapa de calor 3D en el que se vea cómo sube o baja en función de la velocidad si nos resulta abstruso solo con mapa de calor?  
  https://gsyc.urjc.es/jmplaza/students/tfg-deeplearning-mezcla_expertos-juan_simo-2025.pdf
- Sacar histograma por ciudad
- Sacar histogramas de distancia en metros
- Seguir probando con FOV 120 o incluso 150

More details in the [following slides](https://docs.google.com/presentation/d/1zkHQd7Q6C71EKOHBAzPn96qtqaS7fv7c5qVKWE5vmNM/edit?slide=id.g347779b8fc6_0_0#slide=id.g347779b8fc6_0_0)