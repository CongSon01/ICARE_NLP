# ICARE: Cross-Domain Text Classification with Incremental Class-Aware Representation and Distillation Learning

## Overview
The ICARE framework addresses the challenges of natural language inference (NLI) systems, specifically the issues of catastrophic forgetting and overfitting when training on multiple tasks. This framework incorporates an instance selection module and a knowledge distillation method, allowing for effective knowledge transfer across various tasks. Our research demonstrates that ICARE significantly enhances target accuracy and reduces forgetting metrics in text classification tasks.

### Downloading and Pre-processing the data
[Data](https://drive.google.com/drive/folders/1-GpfNkpPcMx7PHyhg_zUCNd5Qgwi9vMl?usp=sharing)

### Training RnD model (SSD Protocol)
```
python runIL_full.py --env RnD_6step
```

### Training RnD model (MSD Protocol)
```
python runIL_full.py --env RnD_3step
```