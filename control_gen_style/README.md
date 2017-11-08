# README #

* Training:
  - Run `pretrain_vae.sh` to pretrain the base VAE
    - An example pretrained VAE is provided: `example_models/pre_vae` 
  - Run `pretrain_disc.sh` to pretrain the discriminator
    - An example pretrained discriminator is provided: `example_models/pre_disc`
  - Run `train_style.sh` to train the full model (for style transfer)
    - An example trained model is provided: `example_models/model`

  - Note: config the hyperparameters in each `*.sh` according to your environment

* Results:
  - Run `python parse_samples.py recon_samples_[x].txt` to parse generate samples

  - The parsed results are in the following format:

    Line 1: orginial sentence 1
    Line 2: the sentiment modified sentence with negative sentiment
    Line 3: the sentiment modified sentence with positive sentiment
    Line 4: orginial sentence 2
    Line 5: the sentiment modified sentence with negative sentiment
    Line 6: the sentiment modified sentence with positive sentiment
    ...
