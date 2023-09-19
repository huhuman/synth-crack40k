# Requirements for Parametric Design of Physics-based Synthetic Data Generation for Learning and Inference of Defect Conditions
The paper was presented in the CRC 2024 conference @ Des Moines, Iowa.

## Implmentation

To implement the work, please follow the below instruction step by step:

### Installation

1. Download the dataset for synthetic crack images ([google drive](https://drive.google.com/drive/folders/1wc5jiEJ1cVaDnWYtZxB0zPIdWyUE-7Hv?usp=sharing))

    This is the generated synthetic crack images used in the training experiments. Note that there are different sets of crack images using different strategies for placing cameras (dome vs. grid)

    Unzip any of them, you will find the data folder with a structure as follows:
    ```
    |--- dataset_dome
       |--- image
          |--- 1_1_01.png
          |--- 1_1_02.png
          |--- ...
       |--- mask
          |--- 1_1_01.png
          |--- 1_1_02.png
          |--- ...
    ```

    The mask images represent the background and crack as the color black and white, respectively

2. Download the publicly available real crack images for validation
    * [CrackForest (Shi et al. 2016)](https://github.com/cuilimeng/CrackForest-dataset/tree/master)
    * [Crack500 (Yang et al. 2019)](https://www.kaggle.com/datasets/pauldavid22/crack50020220509t090436z001)
    * [DeepCrack (Liu et al. 2019)](https://github.com/yhlleo/DeepCrack)
    * [CrackSeg9k (Kulkarni et al. 2022)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGIEBY)

3. Install required Python libraries
    ```
    pip install -r requirement.txt
    ```

### Model training and evaluation
1. Train the model with default setting
    ```
    python segformer_train.py 
    ```
2. 

## Citation
If you find our dataset, code, or paper useful, please cite
```bibtex
@inproceedings{},
  title     = {},
  author    = {},
  booktitle = {},
  year      = {}
```