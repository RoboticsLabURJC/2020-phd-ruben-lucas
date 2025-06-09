---
title: "Carla follow lane DDPG Vs PPO Vs SAC [June 1st]"
excerpt: "Fine tuning perception and training to support Town01 (harder curves)"

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

Objetivos de esta semana:
1. Basar percepción en calzada en lugar de en líneas del carril.
2. Aplicar regresión lineal para calcular el centro (actualmente no funciona sin calcularlo).
3. Cambiar a Carla 0.9.15 y comprobar cuánto tarda con la configuración óptima.
4. Arreglar problema con el cambio de ciudades.
   - El problema parece ser un memory leak al cambiar de ciudades en Carla, que no libera la memoria alocada para el mundo (librerías libopengl de UE) y termina provocando un fallo. Se ha implementado una solución que reinicia el simulador y se vuelve a anclar a él, lo que es más lento pero evita el reinicio cada media hora.
5. Rehacer entrenamiento en:
   - Town01  
   - Town03  
   - Town04  
   - Todos: pendiente solucionar bug para poder relanzar simulaciones.


More details in the [following slides](https://docs.google.com/presentation/d/1lh4_4ZIPLINQvSSTG7BcT-jnNJYoj3cIVT-_18krEww/edit?slide=id.g347779b8fc6_0_0#slide=id.g347779b8fc6_0_0)