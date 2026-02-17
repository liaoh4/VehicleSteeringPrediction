# VehicleSteeringPrediction
Architected an optimized CNN for steering angle regression, utilizing a shallow 3-layer convolutional design to improve generalization on low-resolution imagery.
## Standard PilotNet
A classic 5-layer PilotNet model was implemented. To address the data imbalance, i initially experimented with WeightedRandomSampler to increase the frequency of extreme steering angles and applied Cutout augmentation to force the model to focus on global road features rather than local artifacts. However, despite these efforts, the model still struggled with high validation error, suggesting that the high parameter count led the network to memorize training-specific background noise—an effect exacerbated by the oversampling of rare instances—rather than learning
generalizable road geometry.
## Optimized Shallow Network
To balance computational efficiency with robust generalization, I designed a custom shallow CNN. This architecture reduces depth compared to the baseline PilotNet to prevent the memorization of high-frequency noise in the 64 x 64 grayscale input.
Detailed Layer Specifications:

Convolutional Layer 1: 24 filters, 5x5 kernel, stride 2

Batch Normalization 1: Standardizes feature maps from Conv1

Convolutional Layer 2: 36 filters, 5x5 kernel, stride 2

Batch Normalization 2: Standardizes feature maps from Conv2

Convolutional Layer 3: 48 filters, 5x5 kernel, stride 1

Batch Normalization 3: Standardizes feature maps from Conv3

Flatten Layer: Converts the 48x9x9 feature map into a 1200-unit vector

Fully Connected Layer 1: 100 hidden units with Dropout (0.5)

Fully Connected Layer 2: 50 hidden units with Dropout (0.5)

Output Layer: 1 unit for steering angle regression

Hidden layers utilize the Exponential Linear Unit (ELU) to ensure robust gradient flow for negative inputs, effectively preventing the 'dying ReLU' problem. The output layer has no activation. This allows the model to directly regress continuous steering values from the learned features, avoiding gradient saturation issues commonly found in Tanh or Sigmoid functions. Huber Loss was introduced as the optimization objective.
