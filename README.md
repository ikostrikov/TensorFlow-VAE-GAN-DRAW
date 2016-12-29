# TF-VAE-GAN-DRAW

TensorFlow implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/pdf/1511.06434.pdf), [Variational Autoencoder (also Deep and Convolutional)](http://arxiv.org/pdf/1312.6114v10.pdf) and [DRAW: A Recurrent Neural Network For Image Generation](http://arxiv.org/pdf/1502.04623v2.pdf).

## Run

VAE/GAN:
```bash
python main.py --working_directory /tmp/gan --model vae
```

DRAW:
```bash
python main-draw.py --working_directory /tmp/gan
```

Deep Convolutional Generative Adversarial Networks produce decent results after 10 epochs using default parameters.

###TODO:
- [ ] More complex data.
- [ ] Add [Adversarial Autoencoder](http://arxiv.org/pdf/1511.05644.pdf)
- [ ] Replace current attention mechanism with Spatial Transformer Layer
