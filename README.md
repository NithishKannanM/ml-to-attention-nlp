# From Classical ML to Attention in NLP  
### Understanding Model Behavior Through Progressive Architectures

## The importance of this project

This project was built to **understand how and why different NLP models behave the way they do**, rather than to chase state-of-the-art accuracy.

Instead of jumping directly to complex architectures, I deliberately progressed through:
- a strong classical baseline,
- an initial deep learning approach,
- and an attention-based model,

analyzing **architectural limitations, training dynamics, and design tradeoffs** at each step.

The focus is on **learning, reasoning, and system design**, not just metrics.

---

## Project progression (thinking-first approach)

### 1️⃣ Classical Baseline — SVM + TF-IDF
I started with a linear SVM using TF-IDF features to establish a strong and interpretable baseline.

**Why this step mattered**
- Classical ML is fast, stable, and competitive for text classification
- It sets a realistic performance reference before using deep learning

**Key insight**
> Deep learning must justify its complexity — it is not automatically better because of it.

---

### 2️⃣ CNN–LSTM — Exploring Sequence Modeling Limits
Next, I implemented a CNN–LSTM architecture:
- CNN layers capture local n-gram patterns
- LSTM models sequential information

**What I observed**
- Performance was similar to the classical baseline
- Training was more sensitive to hyperparameters
- Sentence meaning was compressed into the final LSTM timestep

**Limitation identified**
> Relying on the last hidden state creates an information bottleneck for long or complex text.
> The main problem is the last word has more weights to decide the sentiment. 

This step was intentionally explored to **identify architectural weaknesses**, not to optimize performance.

---

### 3️⃣ Attention–LSTM — Architectural Improvement
To address the sequence compression problem, I introduced an attention mechanism over LSTM hidden states.

**Why attention**
- Allows the model to consider all timesteps
- Learns which words contribute most to sentiment
- Removes dependence on a single final state

**Outcome**
- More stable training
- Slight but consistent improvement in generalization
- Better alignment with how humans interpret sentiment in text

**Core insight**
> Improvements came from a better inductive bias, not from adding complexity.

---

## Results (IMDB validation set)

| Model          |  Accuracy |  F1 Score |
|----------------|-----------|-----------|
| SVM + TF-IDF   |   ~0.88   |   ~0.88   |
| CNN–LSTM       |   ~0.88   |   ~0.88   |
| Attention–LSTM | **~0.89** | **~0.89** |

Training loss and validation accuracy were tracked to analyze convergence behavior and stability rather than tuning blindly.

---

## System design & deployment mindset

This project was structured as a **real inference system**, not just a notebook experiment.

Key design choices:
- Shared preprocessing between training and inference
- Tokenizer and model artifacts persisted and reused
- Models exposed via a REST API using Flask
- Tested using curl/Postman to reflect production-style usage

### Example API usage
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"review": "The movie was amazing", "model": "att_lstm"}'
