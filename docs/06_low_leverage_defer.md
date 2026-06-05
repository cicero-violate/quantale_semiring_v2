# 06 — Low-Leverage / Defer List

These algorithms are not bad. They are simply not the highest leverage for the current project resources and goals.

## Defer Category 1 — Deep Vision

| Group            | Examples                                                                  | Why defer                                                                                       |
|------------------+---------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------|
| CNN classifiers  | LeNet, AlexNet, VGG, GoogLeNet, ResNet, DenseNet, MobileNet, EfficientNet | Useful only if the primary input is image/video. Current target is graph/world/macro structure. |
| Object detection | R-CNN, Fast R-CNN, Faster R-CNN, SSD, YOLO                                | Requires visual data pipeline; not core to macro influence mapping.                             |
| Segmentation     | FCN, U-Net, SegNet, Mask R-CNN, DeepLab                                   | Useful for pixel masks, not current graph execution.                                            |

## Defer Category 2 — Generative Vision / GANs

| Group                  | Examples                                       | Why defer                                                         |
|------------------------+------------------------------------------------+-------------------------------------------------------------------|
| GANs                   | DCGAN, WGAN, CGAN, CycleGAN, StyleGAN, Pix2Pix | Expensive and not aligned with executable macro graph.            |
| Image/video generation | GANs, VAEs, Pix2Pix                            | Low leverage unless the project becomes media-generation focused. |

## Defer Category 3 — Heavy Neural Training

| Group                         | Examples                               | Why defer                                                                        |
|-------------------------------+----------------------------------------+----------------------------------------------------------------------------------|
| Large language model training | GPT, T5, BART, RoBERTa, Transformer-XL | Training is unrealistic with current compute. Use pretrained/API models instead. |
| Large transformer training    | Transformer, Transformer-XL            | Too expensive; use as concept or pretrained component only.                      |
| Deep RL                       | PPO, Actor-Critic, DQN, REINFORCE      | Needs environment, simulator, reward, and many episodes.                         |

## Defer Category 4 — Interesting but Nonessential Architectures

| Group                                 | Examples                                       | Why defer                                                             |
|---------------------------------------+------------------------------------------------+-----------------------------------------------------------------------|
| Historical/specialized neural systems | Hopfield Networks, RBM, DBN, Capsule Networks  | Interesting, but not first-order useful for current execution engine. |
| Autoencoders                          | VAE, Denoising Autoencoder, Sparse Autoencoder | Useful later, but PCA and Isolation Forest are cheaper first.         |
| RNN family                            | RNN, LSTM, GRU, Echo State Network             | Classical time-series + graph propagation gives faster leverage now.  |

## Practical Rule

```text
Do not train large neural systems until the symbolic graph can simulate consequences.
```

## When to Revisit Deferred Algorithms

Revisit these only when at least one condition is true:

- The project has image/video input as a core data stream.
- The graph simulator is stable.
- A reward model exists.
- The system has enough logged trajectories for RL.
- A pretrained model can be used cheaply as a component.
- The algorithm directly improves graph scoring, graph compression, or explanation receipts.
