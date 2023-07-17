# Intel Image Classification

In this project, I wanted to showcase how easily one can leverage the capabilities of transfer learning to train a model for classifying image data of Natural Scenes from around the world. Additionally, thanks to PyTorch Lightning, the data processing and model training process are elegantly and succinctly presented.


## Dataset

Data comes from: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
## Run Locally

Clone the project

```bash
  git clone https://github.com/MateuszKunik/Intel-ImageClassification
```

Go to the project directory

```bash
  cd Intel-ImageClassification
```

Install dependencies

```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
  pip install pytorch-lightning
```

Run the code

```bash
  python train.py
```


## Authors

- [Mateusz Kunik](https://www.github.com/mateuszkunik)
