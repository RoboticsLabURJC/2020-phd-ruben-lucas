---
title: "Carla follow lane DDPG Vs PPO Vs SAC [June 3st]"
excerpt: "Refine behavior for intersections and evaluate agents"

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

## Plan de Evaluación y Mejora de Agentes

### Verificar puntos iniciales por ciudad
- Aproximadamente **200–300** puntos por ciudad

### Episodios por entrenamiento
- **DDPG**: 1,000 episodios
- **SAC**: 2,000 episodios
- **PPO**: 15,000 episodios

### Refinar agentes

- ✅ **SAC y DDPG**: Corrigido problema de rizado y volantazos al frenar
- 🔄 **PPO**: El throttle es muy estático y no frena lo suficiente para lograr una conducción suave — *pendiente de mejora*

### Evaluación cuantitativa de entrenamientos (BM)

- Analizar y comparar los 3 modelos en distintos aspectos:
  - **Velocidad media**
  - **Rizado** (desviación típica)
  - **Histogramas** de velocidad y desviación
  - **Invasión de carril**
  - Evaluación en **varias ciudades**


More details in the [following slides](https://docs.google.com/presentation/d/164aH8eTFWlT4yD9gTSy1J2Mti8WDL2IU9GntEKDx1QU/edit?slide=id.g347779b8fc6_0_0#slide=id.g347779b8fc6_0_0)