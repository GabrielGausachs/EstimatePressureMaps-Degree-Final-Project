# TFG
In this repository we will estimate the pressure distribution from IR images of the human body in bed.

### Introduction

Tenir una manera pràctica de mesurar la pressió que un cos humà fa en un matalàs ha estat durant molt de temps un repte no resolt. Aquesta informació és molt valuosa per a la prevenció de les úlceres per pressió en pacients que estan molt temps en llits hospitalaris. Hi ha moltes propostes per estimar la postura humana al llit fent servir sensors, com ara càmeres RGB, càmeres de profunditat, sensors de pressió i mètodes de fusió de sensors. Els sensors de pressió són cars, propensos a errors i s'han de calibrar. Per tant, estimant el mapa de pressió amb altres sensors tindria molt de sentit. Nosaltres proposem estimar-la a partir de imatges infraroges. Per fer-ho proposem implementar una xarxa UNET i entrenar-la amb el dataset SLP.
