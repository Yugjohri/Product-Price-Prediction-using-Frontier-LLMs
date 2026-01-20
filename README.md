# Product Price Prediction using Frontier LLMs

## Overview
This project explores the use of **frontier Large Language Models (LLMs)** for **numerical regression tasks**, specifically predicting product prices from short textual descriptions.
The project also compares **prompt-only LLM approaches** with a **training-based deep learning baseline**, highlighting the strengths and limitations of LLMs in quantitative reasoning.

---

## Key Concepts & Skills
- LLMs for Regression & Numerical Reasoning  
- Prompt Engineering & Optimization  
- Batched Inference using Frontier Models  
- Model Evaluation & Error Analysis  
- Comparison of Prompt-based vs Trained Models  

---

## Architecture Overview

**Pipeline:**
1. Raw product descriptions are preprocessed using an LLM-based formatter
2. Structured attributes (title, category, brand, features) are extracted
3. Prices are predicted using:
   - Frontier LLMs via prompt-based inference
   - A neural network regression baseline
4. Predictions are evaluated against human-labeled ground truth prices

Key Learnings
Frontier LLMs can reason surprisingly well about prices without explicit training
Prompt design has a significant impact on numerical accuracy
Traditional neural networks still outperform LLMs on pure regression when sufficient data is available
Hybrid approaches offer promising future directions
