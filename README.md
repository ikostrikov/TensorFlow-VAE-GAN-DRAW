# TF-GANs

TensorFlow implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/pdf/1511.06434.pdf) and [Variational Autoencoder (also Deep and Convolutional)](http://arxiv.org/pdf/1312.6114v10.pdf).

## Run

```bash
python main_***.py --working_directory /tmp/gan
```

Deep Convolutional Generative Adversarial Networks produce decent results after 10 epochs using default parameters.

###TODO:
- [ ] More complex data.
- [ ] Add attention.
- [ ] Add [Adversarial Autoencoder](http://arxiv.org/pdf/1511.05644.pdf)
- [ ] Replace current attention mechanism with Spatial Transformer Layer
