# TRACE: Traffic Risk Analysis & Critical-Scenario Evaluation

TRACE is an AI-powered framework for real-time risk assessment, scenario generation, and vehicle control in autonomous driving contexts. 

The primary goal of the system is to accurately analyze traffic situations, enable early detection of critical near-miss incidents (such as cut-ins and abrupt braking), and provide explainable risk evaluations.

## System Architecture

The framework consists of three core components:

1. **Simulation & Data Engine:** Data translation pipeline converting real-world driving data (highD dataset) into the MetaDrive simulation environment, including sensor data reconstruction using diffusion models.

2. **Model Serving Layer:** Modular AI models deployed as microservices (hosted via NVIDIA Triton Server).

3. **Statistics Dashboard:** Real-time visualization of vehicle trajectories, Model Metrics, Danger Metrics based on model output

## Model Overview

| Model / Architecture | Purpose / Domain | Implementation |
| ----- | ----- | ----- |
| **LSTM & GNN** | Traffic flow modeling using graph representations (vehicles = nodes, interactions = edges). Prediction of future trajectories and maneuvers (cut-in / cut-out). | PyTorch, PyTorch Geometric |
| **Transformer (LLM)** | Generating MetaDrive scenario configurations via text prompts | Hugging Face (`transformers`) |
| **Random Forest / XGBoost** | Fast tabular ML inference for risk assessment and explainable metrics (XAI). | Scikit-learn, XGBoost |
| **3D-CNN** | Traffic situation analysis based on visual and spatial sensor observations in MetaDrive. | PyTorch3D, MMDetection3D |
| **Diffusion Models** | Reconstruction of noisy/damaged sensor data and generation of synthetic traffic scenarios. | PyTorch |
| **PPO Agent (RL)** | Reinforcement learning agent trained on model output to control autonomous vehicles safely across critical traffic scenarios. | Stable-Baselines3, TorchRL |

## Tech Stack

* **Language:** Python 3.12
* **Simulation:** [MetaDrive](https://github.com/metadriverse/metadrive)
* **Training Data:** [HighD](https://levelxdata.com/highd-dataset/)
* **Machine Learning:** [PyTorch](https://github.com/pytorch/pytorch), [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [Scikit-learn](https://github.com/scikit-learn/scikit-learn)
* **Serving & APIs:** [FastAPI](https://github.com/fastapi/fastapi), [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
* **Testing & Quality:** [Matplotlib](https://github.com/matplotlib/matplotlib) (Evaluation & Visualization)

## Project Structure

```
TRACE/
├── src/
-------/data_pipeline    # Scripts for preprocessing datasets 
-------/inference        # Scripts to call the trained models
-------/models           # Model definition classes
-------/training         # Defined training loops
├── data/
-------/raw              # Raw datasets 
-------/training_data    # Processed datasets to train models
-------/translated       # Datasets translated from highD to metadrive
├── trained_models/      # Final model weights
├── requirements.txt     # Environment dependencies
└── README.md
```

## Team & Project Context

The **TRACE** project is developed as part of a university project (Completion Target: February 2026) by:

* [Leandro Liuzzo](https://github.com/LeleZss)
* [Raul Erbe](https://github.com/LuarEbre)
* [Nico Rauch](https://github.com/Strawberry749)
* [Kevin Kornmann](https://github.com/Sklettnov)
* [Simon Reese](https://github.com/s-reese04)

**Advisor:** Prof. Dr. Jamal Raiyn
