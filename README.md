# Segmentation_Plus

End-to-end customer segmentation and GenAI persona workflow.

## Project Structure

```text
Segmentation_Plus
│
├── data
│   └── dataset.xlsx
│
├── segplus
│   ├── clustering.py
│   ├── persona_generator.py
│   ├── pipeline.py
│
├── notebook.ipynb
└── requirements.txt
```

## Main Notebook

Use `notebook.ipynb` as the final runnable notebook.
It is synced from `segplus/final_segplus_pipeline.ipynb`.

## Setup

```bash
pip install -r requirements.txt
```

## Run

1. Open `notebook.ipynb`.
2. Run cells top-to-bottom.
3. Outputs are written under `segplus_output/`.
