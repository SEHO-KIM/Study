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
    - A contrastive loss is a function whose value is low when q is similar to its positive key k+ and dissimilar to all other keys (considered negative keys for q)
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
~~~
'''
    Algorithm (Pseudocode of MoCo in a PyTorch-like style)
'''
# f_q, f_k: encoder networks for query and key
# queue: dictionary as a queue of K keys (CxK)
# m: momentum
# t: temperature

f_k.params = f_q.params # initialize
for x in loader: # load a minibatch x with N samples
    x_q = aug(x) # a randomly augmented version
    x_k = aug(x) # another randomly augmented version

    q = f_q.forward(x_q) # queries: NxC
    k = f_k.forward(x_k) # keys: NxC
    k = k.detach() # no gradient to keys

    # positive logits: Nx1
    l_pos = bmm(q.view(N,1,C), k.view(N,C,1))

    # negative logits: NxK
    l_neg = mm(q.view(N,C), queue.view(C,K))

    # logits: Nx(1+K)
    logits = cat([l_pos, l_neg], dim=1)

    # contrastive loss, Eqn.(1)
    labels = zeros(N) # positives are the 0-th
    loss = CrossEntropyLoss(logits/t, labels)

    # SGD update: query network
    loss.backward()
    update(f_q.params)

    # momentum update: key network
    f_k.params = m*f_k.params+(1-m)*f_q.params

    # update dictionary
    enqueue(queue, k) # enqueue the current minibatch
    dequeue(queue) # dequeue the earliest minibatch
~~~
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