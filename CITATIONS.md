# Citations & Acknowledgements

## Datasets

1. **Customer Support on Twitter (Primary)**
   - Source: Kaggle — `thoughtvector/customer-support-on-twitter`
   - URL: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
   - Usage: Subsampled `@AppleSupport` customer/reply thread pairs for RAG corpus and golden evaluation set construction.

2. **Banking77 (Reference only)**
   - Source: Hugging Face — `PolyAI/banking77`
   - URL: https://huggingface.co/datasets/PolyAI/banking77
   - Usage: Referenced for intent taxonomy design patterns; not used for training in this submission.

## Libraries & Tools

- **scikit-learn** — TF-IDF vectorization, Logistic Regression classifier, evaluation metrics
- **pandas / numpy** — Data processing
- **nltk / rouge-score** — Text evaluation utilities
- **rich** — CLI table rendering for evaluation scripts
- **kagglehub** — Optional automated Kaggle dataset download (`scripts/build_corpus_from_kaggle.py`)

## Methodology References

- **RAG (Retrieval-Augmented Generation)** — Lewis et al., 2020. Used for grounding replies in historical brand resolutions.
- **LLM-as-Judge evaluation** — Zheng et al., 2023 ("Judging LLM-as-a-Judge"). Inspired our 4-dimensional rubric; implemented as an offline deterministic rubric for reproducibility.

## AI Coding Assistants

This repository was developed with assistance from Cursor AI coding tools, as permitted by the assignment brief.
