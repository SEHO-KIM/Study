# SimCLR
## A Simple Framework for Contrastive Learning of Visual Representations [[Paper]](https://arxiv.org/abs/2002.05709)
- Contribution
    - Composition of data augmentations plays a critical role in defining effective predictive tasks
    - Introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations
    - Contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning
### 2. Method
#### The Contrastive Learning Framework
- Maximize agreement between differently augmented views of the same data example via a contrastive loss in the latent space
- Major components
    - Stochastic data augmentation module
        - Random crop&resize, random color distortions, and random gaussian blur
        - Resulting in two correlated views of the same example
    - Neural network base encoder: ResNet(the output after the average pooling layer)
    - Neural network projection head
    - Contrastive loss function
#### Training with Large Batch Size
- Global BN
#### Evaluation Protocol
- Dataset and Metrics
- Default setting
### 3. Data Augmentation for Contrastive Representation Learning
- Data augmentation defines predictive tasks
#### Composition of data augmentation operations is crucial for learning good representations
#### Contrastive learning needs stronger data augmentation than supervised learning
### 4. Architectures for Encoder and Head
#### Unsupervised contrastive learning benefits (more) from bigger models
#### A nonlinear projection head improves the representation quality of the layer before it
### 5. Loss Functions and Batch Size
#### Normalized cross entropy loss with adjustable temperature works better than alternatives
#### Contrastive learning benefits (more) from larger batch sizes and longer training
### 6. Comparison with State-of-the-art
- Linear evaluation
- Semi-supervised learning
- Transfer learning