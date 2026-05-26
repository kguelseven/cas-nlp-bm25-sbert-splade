# Semantic Search: Comparing BM25, SBERT and SPLADE

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗-Transformers-FFD21E)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-SBERT-2C8EBB)
![BEIR](https://img.shields.io/badge/Benchmark-BEIR%20%2F%20MS%20MARCO-4B8BBE)
![Model](https://img.shields.io/badge/Model-naver%2Fsplade--v3-orange)
![Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab&logoColor=white)

Project work for the **CAS Deep Learning @ FHNW** (Advanced NLP module). A comparative evaluation of three retrieval paradigms on the BEIR / MS MARCO benchmark.

## Project

How do lexical, dense-neural, and sparse-neural retrieval methods compare on the same IR task and where does each fail?

## Approach

- **Dataset:** BEIR / MS MARCO (dev split), 100 queries · 10'000 corpus documents
- **Methods compared:**
  - **BM25** — lexical baseline (TF-IDF family)
  - **SBERT** — dense bi-encoder, cosine similarity over learned embeddings
  - **SPLADE** — learned sparse retrieval with term expansion via MLM
- **Evaluation:** nDCG, MAP, Recall @ k ∈ {1, 3, 5, 10, 100}
- **Case study:** custom query designed to expose the *vocabulary mismatch problem* (e.g. matching climate/weather or car/automobile) and to inspect SPLADE's expanded terms for explainability.

## Result in one line

BM25 is fast and transparent but lexical-only; SBERT and SPLADE both bridge the vocabulary gap, with SPLADE adding explainability and inverted-index compatibility.

## Running

The notebook requires a GPU (L4 / 24 GB recommended for SPLADE) and a Hugging Face token for the gated `naver/splade-v3` model. Designed for Google Colab.

## License

[MIT](LICENSE) © Korhan Gülseven
