
## A Self-Attentional Neural Architecture for Code Completion with Multi-Task Learning


## Prerequisite

- Python 2.7
- Tensorflow [1.12.0](https://github.com/tensorflow/tensorflow/releases/tag/v1.12.0)

## Data
Python and JavaScript datasets: http://plml.ethz.ch

Java dataset: https://drive.google.com/open?id=1xxnYAu8L5i6TpNpMNDWxSNsOs3XYxS6T

Each program is represented in its AST format, and the AST is serialized in in-order depth-first traversal to produce the AST node sequence.

Data process code is in the "preprocess_code" diretory.

## Train models
```python
python train_gpu_mt.py --alpha ${weight for type prediction loss} --mem_len ${memory length} --model_dir ${path_to_save} 
```
