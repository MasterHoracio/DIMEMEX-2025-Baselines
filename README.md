# DIMEMEX 2025 Baselines

This repository presents the official baselines proposed for the **DIMEMEX 2025** task, which focuses on the detection of hate speech and inappropriate content in memes written in Mexican Spanish.

For more details about the task, please visit the official competition page:  
👉 [https://codalab.lisn.upsaclay.fr/competitions/22012](https://codalab.lisn.upsaclay.fr/competitions/22012)

---

## LLM Baseline (Subtask 3)

The baseline provided for subtask 3 is based on a **zero-shot in-context learning** approach. It uses the `meta-llama/Llama-3.2-11B-Vision-Instruct` model to classify memes into one of the following three categories:

- Hate Speech  
- Inappropriate Content  
- None

The model receives:
- The image of the meme,
- The extracted text via OCR, and
- An automatically generated description of the meme.

All three components are used as input to the model in order to make a classification decision.

### Evaluation Instructions

To run the model, use the `--evaluation_type` argument to specify the evaluation dataset. The available options are:

```bash
python3 llm_baseline.py --evaluation_type validation
python3 llm_baseline.py --evaluation_type test
