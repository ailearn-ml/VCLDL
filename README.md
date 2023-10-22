# Code and Appendix for VCLDL.

Code for "Variational Continuous Label Distribution Learning for Multi-Label Text Classification" in IEEE Transactions on Knowledge and Data Engineering 2023.

## Requirements

- Python >= 3.6
- PyTorch >= 1.10
- NumPy >= 1.13.3
- Scikit-learn >= 0.20

## Running the scripts

To train basic XML-CNN model in the terminal, use:

```bash
$ python run_step1_run_xmlcnn.py
```

To train and test the VCLDL-based XML-CNN model in the terminal, use:

```bash
$ python run_step2_run_vcldl_xmlcnn.py
```

To train and test the Hierarchical VCLDL-based XML-CNN model in the terminal, use:

```bash
$ python run_step2_run_vcldl_xmlcnn_hierarchical.py
```

## Using pre-trained model

Download the [pre-trained XML-CNN model](https://drive.google.com/drive/folders/1pSDj5__3Kps05ZKWHtSWADOxWT5LjP9c?usp=share_link) and put it into the main folder.