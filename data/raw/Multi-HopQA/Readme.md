# HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering

HotpotQA is a large-scale question answering dataset featuring natural, multi-hop questions. It is designed to foster the development of explainable NLP systems by providing strong supervision for supporting facts.

## 📌 Dataset Overview
Unlike standard QA datasets that can often be solved with single-document retrieval, HotpotQA requires models to perform **multi-hop reasoning**—combining information from multiple documents to find the correct answer.

### Key Features:
* **Multi-hop Reasoning**: Questions are specifically designed so they cannot be answered by looking at a single paragraph or sentence.
* **Supporting Facts**: The dataset includes sentence-level annotations for the facts that support the answer, enabling more explainable model outputs.
* **Diverse Content**: Questions cover a wide range of topics sourced from Wikipedia.
* **Explainability**: By requiring models to identify supporting evidence, it moves beyond simple "black-box" prediction.

## 🎓 Research Credits
The dataset was collected and curated by a collaborative team of NLP researchers from:
* Carnegie Mellon University
* Stanford University
* Université de Montréal

## 📄 Citation
If you use this dataset in your research, please cite the following EMNLP 2018 paper:

```bibtex
@inproceedings{yang2018hotpotqa,
  title={{HotpotQA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William W. and Salakhutdinov, Ruslan and Manning, Christopher D.},
  booktitle={Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  year={2018}
}
```