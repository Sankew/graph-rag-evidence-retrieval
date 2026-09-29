# HotpotQA sample attribution and license

The `documents.jsonl` and `questions.jsonl` files are adapted from the HotpotQA distractor development set by Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning (2018).

Source: [HotpotQA project](https://hotpotqa.github.io/) and its [Hugging Face dataset mirror](https://huggingface.co/datasets/hotpotqa/hotpot_qa). The source Parquet SHA-256 is `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`.

Changes: selected 150 development questions by SHA-256 question ID order, balanced 75 bridge and 75 comparison questions, converted their context paragraphs and labels to JSONL, and deduplicated paragraphs by title. The selection and conversion code is in `scripts/build_hotpot_sample.py`.

The included adapted data is shared under [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/). This license covers the HotpotQA-derived sample data, not the project code.
