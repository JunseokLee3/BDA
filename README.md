# BDA
The repository includes **test code** and **train code** related to the research paper "***BDA: Bi-directional Attention for Zero-Shot Learning.***"
## Running Environment

**BDA** is primarily implemented using Python 3.8.8 and [PyTorch](https://pytorch.org/) 1.8.0. To set up all necessary dependencies:
```
$ pip install -r requirements.txt
```
Furthermore, we utilize [Weights & Biases](https://wandb.ai/site) (W&B) to manage and arrange the experimental results. You may need to consult W&B's [online documentation]
(https://docs.wandb.ai/quickstart) for a quick start. To execute this code, you can either [register](https://app.wandb.ai/login?signup=true) for an online account to monitor experiments.

## Preparing Dataset and Model
The model was trained on three widely-used ZSL benchmarks: [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html), and [AWA2](http://cvml.ist.ac.at/AwA2/), adhering to the [xlsa17](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip) data split. To test the **BDA**, first download these datasets along with xlsa17. Next, unzip and arrange them in the following manner.
We also offer pre-trained models (available on [Google Drive](https://drive.google.com/drive/folders/18Hr24iSPqb1oZGs6j7XT-y-VZZo3WWyI?usp=sharing)) for three distinct datasets: CUB, SUN, and AWA2 in the CZSL/GZSL context. Download the model files and their respective datasets, then arrange them in the following way:
``` 
. 
├── data 
│   ├── CUB/CUB_200_2011/... 
│   ├── SUN/images/... 
│   ├── AWA2/Animals_with_Attributes2/... 
│   └── xlsa17/data/... 
├── model 
│   └── chekpoint       
│       ├── CUB_CZSL.pth 
│       ├── CUB_GZSL.pth 
│       ├── SUN_CZSL.pth 
│       ├── SUN_GZSL.pth 
│       ├── AWA2_CZSL.pth 
│       └── AWA2_GZSL.pth 
└── ··· 
```

## Visual Features Preprocessing
At this stage, execute the commands below to extract the visual features from the three datasets:

```
$ python preprocessing.py --dataset CUB --compression --device cuda:0
$ python preprocessing.py --dataset SUN --compression --device cuda:0
$ python preprocessing.py --dataset AWA2 --compression --device cuda:0
```

```
$ python train_BDA_cub_sweep_GZSL.py   # CUB GZSL
$ python train_BDA_cub_sweep_CZSL.py   # CUB CZSL
$ python train_BDA_sun_sweep_GZSL.py   # SUN GZSL
$ python train_BDA_sun_sweep_CZSL.py   # SUN CZSL
$ python train_BDA_awa2_sweep_GZSL.py  # AWA2 GZSL
$ python train_BDA_awa2_sweep_CZSL.py  # AWA2 CZSL
```

```
$ python test_cub_gzsl.ipynb   # CUB GZSL
$ python test_cub_czsl.ipynb   # CUB CZSL
$ python test_sun_gzsl.ipynb   # SUN GZSL
$ python test_sun_czsl.ipynb   # SUN CZSL
$ python test_awa2_gzsl.ipynb  # AWA2 GZSL
$ python test_awa2_czsl.ipynb  # AWA2 CZSL
```
**Note**: 
Make sure to load the appropriate configuration when targeting the CZSL task.

## Results
Outcomes from our published models, utilizing different evaluation methods across three datasets, are presented in both conventional ZSL (CZSL) and generalized ZSL (GZSL) configurations.
| Dataset | Acc(CZSL) | U(GZSL) | S(GZSL) | H(GZSL) |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| CUB | 78.0 | 69.5 | 69.8 | 69.7 |
| SUN | 66.6 | 55.2 | 32.7 | 41.1 |
| AWA2 | 70.0 | 62.9 | 79.7 | 70.3 |
## References
Parts of our codes based on:
* [shiming-chen/TransZero](https://github.com/shiming-chen/TransZero)
* [hbdat/cvpr20_DAZLE](https://github.com/hbdat/cvpr20_DAZLE)
* [zhangxuying1004/RSTNet](https://github.com/zhangxuying1004/RSTNet)
## Contact
If you have any questions about codes, please don't hesitate to contact us by 2021126776@dgu.ac.kr
