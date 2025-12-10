# Multimodal AI for Chest X-Ray Diagnosis

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
Medical diagnosis often requires synthesizing information from multiple sources. This project implements a **Multimodal Deep Learning System** that fuses **Computer Vision** (Chest X-Rays) and **Natural Language Processing** (Radiology Reports) to predict thoracic diseases.

The model mimics a clinician's workflow by integrating visual findings with textual impressions, achieving significantly higher accuracy than unimodal baselines.

## 🚀 Key Features
* [cite_start]**Multimodal Architecture:** Fuses image features (DenseNet-121) and text features (ClinicalBERT)[cite: 19, 20].
* [cite_start]**Advanced Training:** Implements differential learning rates and a freeze/unfreeze strategy for backbone fine-tuning[cite: 22, 25].
* [cite_start]**Interactive Web App:** Deployed using **Streamlit** for real-time inference[cite: 61].
* [cite_start]**High Performance:** Achieved an **Overall Macro AUC of 0.9281** on the unseen test set[cite: 29].

## 📊 Results & Metrics
The model was evaluated on the Indiana University Chest X-Ray Collection.

| Disease Class | Test AUC | Performance |
| :--- | :--- | :--- |
| **Pneumonia** | **0.9647** | [cite_start]Exceptional detection of visual/textual features[cite: 32]. |
| **Edema** | **0.9548** | [cite_start]High confidence in identifying fluid buildup[cite: 32]. |
| **Cardiomegaly** | **0.9338** | [cite_start]Effective at identifying enlarged heart conditions[cite: 32]. |
| **Atelectasis** | **0.9368** | [cite_start]Strong performance in lung collapse detection[cite: 32]. |
| **Consolidation**| 0.8503 | [cite_start]Good performance given the ambiguity of the condition[cite: 32]. |

## 🛠️ Tech Stack
* **Image Backbone:** `DenseNet-121` (Pre-trained on ImageNet)
* **Text Backbone:** `ClinicalBERT` (Pre-trained on MIMIC-III clinical notes)
* **Frameworks:** PyTorch, Transformers (Hugging Face), Torchvision
* **Deployment:** Streamlit, Ngrok

## 📂 Project Structure
```bash
├── src/                # Core model definitions and training scripts
│   ├── image_model.py      # DenseNet implementation
│   ├── text_model.py       # ClinicalBERT wrapper
│   └── multimodal_model.py # Late fusion network
├── app/                # Streamlit web application
└── notebooks/          # Exploratory Data Analysis (EDA)
