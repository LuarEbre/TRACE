# Sprint Backlog 3 (13/11/2026)

## Update Aktueller Stand

Simon:
- GPU activate im tranining script
- Translate script zu metadrive
Raul:
- Nix neues
Leandro:
- Requirements probleme (windows, python version)
Kevin:
- Model research, sonst nix neues
Nico:
- Probleme bei dependencies, sonst nix neues

## Welche Python Version?
- 3.12, außer es kommen weitere probleme

## Fragen zu dependencies und requirements.txt
- Zwei requirements (windows, linux) da große Unterschiede
- Absofort alle (windows, linux) nutzen die gleichen dependencies

## Welche Outputs sollen die models haben?
GNN: X'Y' vom nächsten frame (Trajektorenanalyse) 
LSTM: Predicteter TTC pro Auto in den nächsten Frames
CNN: Anomalie erkennen (Continious Anomalie Wert [0,1])
RF/XGBoost: Discrete States erkennen (Überholt, Fährt normal)
Diffusion: Parameter die fehlen
LLM: Config für metadrive

## Was fehlt noch im Sprint?
- Translation Script überprüfen (Was fehlt noch von highD -> metadrive)
- Pro Model: Preprocessing der translated csv files
=> Training datasets pro model

## Aufgabenteilung
- Translation Script: Leandro
- Pro Model Preprocessing: Jeder für sein model

## Nächstes Sprint Meeting
- Freitag 20.11.2026
