# Research on Low Light Image Enhancement Based on Deep Learning

![Python version](https://img.shields.io/badge/Python-3.10+-blue) ![Pytorch version](https://img.shields.io/badge/Pytorch-2.3.0+-blue) ![CUDA version](https://img.shields.io/badge/CUDA-12.1+-blue)

Single GPU training | [Multiple GPUs training](https://github.com/JohnScotttt/GraduationDesign/tree/multiple/README.md)

## Quickly Start

### Clone repository

```cmd
git clone -b single https://github.com/JohnScotttt/GraduationDesign.git
```

### Create new Python environment

You can use Conda or Venv to create a new clean Python environment to ensure that the program runs correctly. 

**We strongly recommend that you do so!🥰**

### Install Pytorch

We recommend that you install PyTorch version 2.3.0 or higher, and ensure that the CUDA version is greater than 12.1.

You can check [PyTorch's official website](https://pytorch.org/get-started/previous-versions/) for installation guides.

### Install requirements

```cmd
cd GraduationDesign
pip install -r requirements.txt
```

### Generate datalist

You can add the following code to your Python file:

```python
from utils.preprocessing import generate_datalist
generate_datalist(dataset_path, dataset_name, output_dir)
```

This will generate a TSV file with the paths to the dataset images.

### Train with default config

```cmd
python train.py
```

You need to modify the **`train_tsv_file`** and **`eval_tsv_file`** in **cfg/default.toml**.

### Enhance low-light images using pre-trained model

You can add the following code to your Python file:

```python
from modules.runner import eval
eval(config_file, model_path, save_images)
```

## Train

The key training parameters are set in **cfg/default.toml**, you can create a custom TOML file to tailor the training process.

You only need to specify the parameters you want to update in your custom TOML file, and the rest will automatically use the default configurations. Such as:

```toml
[settings]
	epochs = 100
	name = "LOL"
[diffusion]
	num_diffusion_timesteps = 1000
```

Training depends on a dataset TSV file, which can be generated using the **_generate_datalist_** function in **utils/preprocessing.py**.

```
generate_datalist(dataset_path, dataset_name, output_dir)
```

Currently, only the LOLv1 and LOLv2 datasets are supported:

- Set **`dataset_name="LOL"`** for the LOLv1 dataset.
- Set **`dataset_name="LOLv2"`** for the LOLv2 dataset.
