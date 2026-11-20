# Sprint Backlog 4 (20/11/2026)

## Update Aktueller Stand
Simon: Preprocessing fertig / erstes script ist da. 

Kevin: Diffusion Update: MLP model, daten vorbereitet für model, Output = TTC

Raul: Translation Script steht (String fehler noch), Output = Anomalies tracken

Leandro: Script, dass alle parameter von metadrive per frame gibt, 
Model ist potentiell Qwen2.5:3b, Google Collab für fine tuning (QLORA) aufgesetzt

Nico: Translation Script mitgeholfen, Probleme gefunden: Alle fahrzeuge in frame 1 haben gleiche id

## Ordnerstruktur (Leere dir mit pushen)
Geeinigt auf eine Ordnerstruktur

## Was fehlt von Sprint 2?
- Translation script muss optimiert werden
- Preprocessing: RF, CNN, Diffusion, 

## Aufgaben Sprint 3
- Model Prototyp für jedes model
- Diagrame, für offline und online

## Entscheidung:
LSTM fürs erste nicht machen (Diffusion übernimmt TTC prediction)

## Aufgabenverteilung
- Jeder sollte am Model definition und training arbeiten
- Simon: Live diagram machen und Live preprocessing script
- Leandro und Raul: Anomalie mit LLM gen abgleichen

## Nächstes Sprint Meeting
- 27.11.2026 (Update Meeting)
