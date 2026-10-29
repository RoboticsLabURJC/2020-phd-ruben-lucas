---
title: Addressing perception robustness and inference latency in lead-vehicle ACC scenarios (October week 3 - week 4)
excerpt: yolopv2 in inference
---

Expected goals:

- Mejora percepción
- Afinar homografía y elección de las lineas para hacerlo sobre BEV en lugar de 2D front camera. Facilita y hace más robusta esta decisión
- Intento de reentrenar aún más YOLOP sin mucho éxito
- Entrenado BEVFormer con apollo. Si los pasos anteriores no mejoran el rendimiento del ACC. Exploraré integrar este modelo para obtener BEV directamente de la camara
- Reentrenando followlane para confirmar que los ajustes mejoran lo presente
- Entrenando no solo en town03, lo cual hacía que no generalizase nada bien
- En lugar de usar 1 solo. Usamos 04, 01 y 03 pero cambiamos de ciudad cada mucho tiempo para minimizar CARLA server OOMs y reinicio de entrenamiento desde el último checkpoint
- Ajustes en BM para poder lanzar ACC en modo síncrono y con GUI (parece que la percepción es más lenta y esto unido a ligeras discrepancias en como funciona BM y RL hace que el comportamiento empeore al llevar el agente de RLStudio a BM)
- Pendiente rebajar tiempo de percepción de 110ms a 80ms
- Pendiente afinar ligera discrepancia entre BM y RLStudio (E.ROS delay)

More details in [this link](https://docs.google.com/presentation/d/1O0sLfCiAm29LiHJln1CSHf3Il-ws_7b-3WhtWMFSQOc/edit).
