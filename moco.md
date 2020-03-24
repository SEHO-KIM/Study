# MoCo
## Momentum Contrast for Unsupervised Visual Representation Learning

### version 1 [[Paper]](https://arxiv.org/abs/1911.05722)
- Contribution
    - Enable building a large and consistent dictionary on-the-fly that facilitates contrastive unsupervised learning
    - Outperform its supervised pre-training counterpart in 7 detection/segmentation tasks on PASCAL VOC, COCO, and other datasets, sometimes surpassing it by large margins
    - The gap between unsupervised and supervised representation learning has been largely closed
#### 3. Method
##### Contrastive Learning as Dictionary Look-up
- Dictionary Look-up
    - Keys {k0, k1, k2,...}
- q = f_q(x^q) where f_q is encoder network and x^q is query sample which can be images, patches... 
- Contrastive loss
    - If q(query) and positive key k+ are similar, loss value is low
    - Measured by dot product, called InfoNCE
##### Momentum Contrast
- Dictionary as a queue
    - Decouple the dictionary size from the mini-batch size(flexible like hyper-parameter)
    - The current mini-batch is enqueued to the dictionary, and the oldest mini-batch in the queue is removed
    - Represent a sampled subset of all data
- Momentum update
    - θ_k <- mθ_k + (1-m)θ_q
- Relations to previous mechanisms
    - end-to-end
    - memory bank
    - MoCo
##### Pretext Task
- Technical details
    - ResNet as the encoder(output vector is normalized by its L2-norm)
    - The temperature τ is set as 0.07
- Shuffling BN
#### 4. Experiments
- Dataset
    - ImageNet-1M(IN-1M), Instagram-1B(IG-1B)
- Training
    - ResNet50
    - SGD optimizer (weight decay 0.0001, momentum 0.9)
    - IN-1M
        - Batch size: 256 in 8 GPUs
        - Initial learning rate: 0.03
        - Learning step: 120, 160 (total 200 epochs, learning decay 0.1)
    - IG-1B
        - Batch size: 1024 in 64 GPUs
        - Initial learning rate: 0.12
        - Learning step: exponentially decayed by 0.9x after every 62.5k iterations (64M images)
##### Linear Classification Protocol
##### Transferring Features

---
### version 2 [[Tech Report]](https://arxiv.org/abs/2003.04297)
- Verify the effectiveness of two of SimCLR’s design improvements by implementing them in the MoCo framework
#### 3. Experiments
- Settings
- MLP head
    - Replace the fc head in MoCo with a 2-layer MLP head(hidden layer 2048-d, with ReLU)
- Augmentation
    - Original augmentation + blur augmentation
- Comparison with SimCLR
    - Outperform SimCLR