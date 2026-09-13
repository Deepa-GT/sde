# Citations, References & Acknowledgements

<div align="center">

**Hiver SDE Intern Take-Home Assessment**  
Project: Apple Support AI Agent & Evaluation Harness  
Repository: [https://github.com/Deepa-GT/sde](https://github.com/Deepa-GT/sde)  

</div>

---

## 1. Datasets & Corpora

### Primary Dataset: Customer Support on Twitter
- **Citation**: Thoughtvector. (2017). *Customer Support on Twitter*. Kaggle.
- **URL**: [https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- **Application in System**: Filtered and structured multi-turn conversation threads specifically directed to and from `@AppleSupport` to build the historical resolution RAG index (`data/raw_support_tweets.csv`) and sample authentic queries for the Golden Evaluation Set (`data/golden_eval_set.json`).
- **BibTeX**:
```bibtex
@misc{thoughtvector_twitter_support_2017,
  author = {Thoughtvector},
  title = {Customer Support on Twitter: Over 3 Million Tweets and Responses from Top Brands},
  year = {2017},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter}
}
```

### Secondary Reference: Banking77
- **Citation**: Casanueva, I., et al. (2020). *Efficient Intent Detection with Dual Sentence Encoders*. Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI.
- **URL**: [https://huggingface.co/datasets/PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
- **Application in System**: Referenced during the exploratory phase to evaluate intent granularity trade-offs; informed our decision to consolidate into a focused 7-class business taxonomy (see `DECISION_LOG.md` ADR-02).
- **BibTeX**:
```bibtex
@inproceedings{casanueva2020efficient,
  title={Efficient Intent Detection with Dual Sentence Encoders},
  author={Casanueva, I{\~n}igo and Tem{\v{c}}inas, Tadas and Gerz, Daniela and Henderson, Matthew and Vuli{\'c}, Ivan},
  booktitle={Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI},
  pages={38--45},
  year={2020}
}
```

---

## 2. Core Methodology References

### Retrieval-Augmented Generation (RAG)
- **Citation**: Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems (NeurIPS 2020), 33, 9459-9474.
- **Application in System**: Conceptual foundation for grounding agent responses in historical `@AppleSupport` resolutions to eliminate policy hallucination.
- **BibTeX**:
```bibtex
@article{lewis2020retrieval,
  title={Retrieval-augmented generation for knowledge-intensive nlp tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and others},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={9459--9474},
  year={2020}
}
```

### LLM-as-a-Judge Evaluation & Calibration
- **Citation**: Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. Advances in Neural Information Processing Systems (NeurIPS 2023), 36.
- **Application in System**: Inspired our 4-dimensional evaluation rubric (Correctness, Grounding, Tone, Policy) and the statistical Pearson calibration protocol against human annotations (`scripts/calibrate_judge.py`).
- **BibTeX**:
```bibtex
@article{zheng2023judging,
  title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},
  author={Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and Tian, Siyuan and Wu, Hao and Zhang, Zi and Li, Ekan and Chen, Beidi and Xing, Eric and Stoica, Ion and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}
```

---

## 3. Open Source Software & Libraries

The following open-source frameworks were utilized in this implementation:
- **Scikit-Learn**: Vectorization, metrics computation, and Logistic Regression baseline (Pedregosa et al., JMLR 2011).
- **Pandas & NumPy**: Data processing and statistical calibration routines.
- **ROUGE-Score & NLTK**: Longest Common Subsequence (ROUGE-L) and n-gram overlap evaluations.
- **Streamlit & Plotly**: Interactive web dashboard and reactive metric visualization.
- **Rich**: Terminal console rendering for benchmark scripts.

---

## 4. AI Coding Assistants Disclosure

In full accordance with the assignment guidelines (*"You may use AI coding assistants freely. We will ask you to explain and modify your own code live"*):
- AI coding assistants were leveraged for boilerplate generation, CSS styling, and Markdown document drafting.
- All architectural decisions, mathematical rubric formulations, intent taxonomy schemas, and evaluation logic were directed and verified by the author.
