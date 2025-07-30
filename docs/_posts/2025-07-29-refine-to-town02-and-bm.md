---
title: "Carla follow lane DDPG Vs PPO Vs SAC [July 3rd]"
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

- Linearizar `x, y` para mapearlo a una única coordenada **“l”**. Podemos utilizar la trayectoria ideal.
- Mostrar en el eje **x** si son curvas pronunciadas, suaves o rectas.
- Probar la velocidad más baja con FOV más razonable (e.g. 90 - 110).
- Reducir aún más la velocidad ayuda más que ampliar el FOV, pero sigue fallando en algunas.
- **Finalmente** se observó que el trazado obliga a hacer un giro abrupto, dificultando algunas curvas. **Solucionado reubicando la cámara.**
- Rebalanceando dataset para que no dé tanta importancia a rectas y curvas suaves.
- Castigando más el **“salirse”**.
- Relajando la condición que establece que si no estás cerca de la línea, no te premio nada, para que en curvas sacrifique un poco en pos de reducir el riesgo a salirse.
- Sacar todas las métricas de **BM** para revisar.
- Ajustar cosas como distancia en metros.
- Histogramas por ciudad, no por experimento.

More details in the [following slides](https://docs.google.com/presentation/d/1P0TOuDb0i-igDBTC0QhF9PJ-wjylsEMKk9zJPUfp_Es/edit?slide=id.g347779b8fc6_0_0#slide=id.g347779b8fc6_0_0)