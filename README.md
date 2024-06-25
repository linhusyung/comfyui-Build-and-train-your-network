# ComfyUI Build and Train Your Network

Stable Diffusion is an image generation technique based on diffusion models. Its core idea involves simulating diffusion
processes by iteratively adding noise and using neural networks to predict and remove the noise, thereby generating
high-quality images. This approach is not limited to image generation; with appropriate network architecture and
training data, it can be adapted for various other tasks. The application of neural networks extends beyond image
generation. By adjusting network structures and loss functions, neural networks can also perform tasks such as
classification and regression. This flexibility makes neural networks a powerful tool for handling a wide range of
machine learning tasks.

This project aims to expand custom neural network layers (such as linear layers, convolutional layers, etc.) within
ComfyUI and provide simplified task training functionalities. Through this project, users can easily construct custom
neural network layers and perform training in ComfyUI using a graphical interface.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Description](#Description)

## Features

- ✨ **Custom Neural Network Layers**: Users can add and configure custom neural network layers, such as linear layers
  and convolutional layers, within ComfyUI.
- 🚀 **Simplified Task Training Functionality**: It offers a streamlined training process.

## Installation

### Steps

1. **Clone the repository**
    ```bash
    git clone https://github.com/linhusyung/comfyui-Build-and-train-your-network.git
    ```

2. **Navigate to the project directory**
    ```bash
    cd comfyui-Build-and-train-your-network
    ```

3. **Modify ComfyUI for gradient calculation**

   In ComfyUI, tensor calculations do not produce gradients. To resolve this issue, you need to modify the
   `ComfyUI/execution.py` file by commenting out `with torch.inference_mode():` and moving all the code within this
   context manager one level forward. If you prefer not to do this manually, you can copy the `replace/execution.py`
   from the project to
   `ComfyUI/execution.py`, like this:

    ```bash
    copy .\replace\execution.py ..\..\execution.py
    ```

   **Note:** Modifying `ComfyUI/execution.py` might cause issues with image generation. If this happens, don't worry;
   you can restore the original state by running the following command:

    ```bash
    copy .\replace\execution1.py ..\..\execution.py
    ```

## Description

### Conv Layer

<div style="text-align:center;">
    <img src="readme_fig/cnn_layer/cnn_layer.png" alt="Conv Layer" width="400"/>
</div>

**Description:**
The operation of a convolutional layer involves sliding convolution kernels over the input feature map, performing
element-wise multiplication with the local region at each step, and then summing all the multiplication results to form
a single element in the output feature map. The movement of the kernels and the size of the output feature map are
controlled by setting the stride and padding. If the input feature map has multiple channels (e.g., three channels for
an RGB image), each channel of the input is multiplied element-wise with the corresponding channel of the kernel, and
the results across all channels are summed to produce the output. When the number of output channels is 8, the system
employs 8 convolution kernels for convolutional operations and concatenates them along the channel dimension to form the
output feature map.

### Activation Function

<div style="text-align:center;">
    <img src="readme_fig/activation_function/activation_function.png" alt="Activation Function" width="400"/>
</div>

**Description:**
The activation function plays a crucial role in neural networks by introducing non-linearity, allowing the network to
learn complex patterns and representations. The activation function operates on the output of each neuron, transforming
the linear combination of inputs into a non-linear output. Common activation functions include:

#### ReLU (Rectified Linear Unit)

ReLU transforms negative input values to zero and keeps positive values unchanged, defined as:
$f(x) = \max(0, x)$
ReLU is computationally simple and widely used in many deep learning models.

#### Sigmoid

The Sigmoid function compresses input values to a range between 0 and 1, defined as:
$f(x) = \frac{1}{1 + e^{-x}}$
It is often used in the output layer to represent probabilities.

#### Tanh (Hyperbolic Tangent)

Tanh compresses input values to a range between -1 and 1, defined as:
$f(x) = \tanh(x)$
Compared to Sigmoid, Tanh's output mean is zero, which can be beneficial for certain networks.

#### Leaky ReLU

Leaky ReLU is a variant of ReLU that allows small negative values to pass through, defined as:
$f(x) = x \text{ if } x > 0, \text{ otherwise } f(x) = \alpha x$
(usually $\alpha = 0.01$).

#### Softmax

The Softmax function converts a vector of values into a probability distribution, where the probability of each value is
proportional to the exponent of the value, defined as:
$f(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$
Softmax is often used in the output layer of a classification network to represent the probabilities of each class.

By applying an activation function after each layer, neural networks can capture more complex and diverse features,
thereby enhancing the model's expressive power and performance.

### Normalization

<div style="text-align:center;">
    <img src="readme_fig/normalization/normalization.png" alt="Normalization" width="400"/>
</div>

**Description:**
Normalization techniques play a crucial role in improving the training and performance of neural networks by stabilizing
and accelerating convergence. Here are different types of normalization methods commonly used in deep learning:

#### Batch Normalization (BatchNorm1d, BatchNorm2d)

Batch Normalization normalizes the activations of each layer across the batch dimension. For 1-dimensional and
2-dimensional inputs:

- **BatchNorm1d**: Normalizes along the feature dimension (typically used in fully connected layers).
- **BatchNorm2d**: Normalizes along the channel dimension (typically used in convolutional layers).

For both BatchNorm1d and BatchNorm2d, the normalization process is defined as:
$f(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$
where $\( \mu \) and \( \sigma^2 \)$ are the mean and variance of the batch, $\( \epsilon \)$ is a small constant to
prevent
division by zero, $\( \gamma \) and \( \beta \)$ are learnable parameters (scale and shift).

#### Layer Normalization (LayerNorm)

Layer Normalization normalizes the activations across the feature dimension independently for each example in the batch.
It is defined as:
$f(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta $
where $\( \mu \) and \( \sigma^2 \)$ are the mean and variance computed along the feature dimension.

By incorporating normalization techniques like BatchNorm1d, BatchNorm2d, or LayerNorm into neural networks, training can
be more stable and converge faster, leading to improved overall performance and generalization.

### Pooling Layer

<div style="text-align:center;">
    <img src="readme_fig/pooling_layer/pooling_layer.png" alt="Pooling Layer" width="400"/>
</div>

**Description:**
Pooling layers are used in convolutional neural networks (CNNs) to progressively reduce the spatial size of the
representation to reduce the amount of parameters and computation in the network, and to control overfitting. Here are
different types of pooling layers commonly used:

#### Adaptive Average Pooling (AdaptiveAvgPool2d)

Adaptive Average Pooling allows the output size to be specified instead of the kernel size. It adaptively performs
average pooling to the desired output size. For 2-dimensional inputs:

- **AdaptiveAvgPool2d**: Outputs a 2D tensor with a specified output size.

The adaptive average pooling operation is defined as:
$$
f(x)_{i,j} = \frac{1}{\text{kernel\_size}[0] \times \text{kernel\_size}[1]} \sum_{m=0}^{\text{kernel\_size}[0]-1} \sum_
{n=0}^{\text{kernel\_size}[1]-1} x_{i \times \text{stride}[0] + m, j \times \text{stride}[1] + n}
$$

#### Max Pooling (MaxPool2d)

Max Pooling extracts the maximum value from each patch of the feature map. For 2-dimensional inputs:

- **MaxPool2d**: Computes the maximum value over a spatial window.

The max pooling operation is defined as:
$$
f(x)_{i,j} = \max_{m=0, \ldots, \text{kernel\_size}[0]-1} \max_{n=0, \ldots, \text{kernel\_size}[1]-1} x_{i \times
\text{stride}[0] + m, j \times \text{stride}[1] + n}
$$

#### Average Pooling (AvgPool2d)

Average Pooling computes the average value of each patch of the feature map. For 2-dimensional inputs:

- **AvgPool2d**: Computes the average value over a spatial window.

The average pooling operation is defined as:
$$
f(x)_{i,j} = \frac{1}{\text{kernel\_size}[0] \times \text{kernel\_size}[1]} \sum_{m=0}^{\text{kernel\_size}[0]-1} \sum_
{n=0}^{\text{kernel\_size}[1]-1} x_{i \times \text{stride}[0] + m, j \times \text{stride}[1] + n}
$$

Pooling layers help in reducing the spatial dimensions of the input while retaining important features, making the
network more robust and efficient in processing spatial hierarchies.

### View

<div style="text-align:center;">
    <img src="readme_fig/view.png" alt="View" width="400"/>
</div>

**Description:**
pass

### Fully Connected Layer

<div style="text-align:center;">
    <img src="readme_fig/linear/linear.png" alt="Fully Connected Layer" width="400"/>
</div>

**Description:**
pass

### Create model

<div style="text-align:center;">
    <img src="readme_fig/create_model.png" alt="Create model" width="400"/>
</div>

**Description:**
pass

### create dataset

<div style="text-align:center;">
    <img src="readme_fig/dataset/create_dataset.png" alt="create dataset" width="400"/>
</div>

**Description:**
pass

### train

<div style="text-align:center;">
    <img src="readme_fig/train.png" alt="train" width="400"/>
</div>

**Description:**
pass

### pre train model

<div style="text-align:center;">
    <img src="readme_fig/pre_train/pre_train.png" alt="pre train model" width="400"/>
</div>

**Description:**
pass

### Example of Residual Module

<div style="text-align:center;">
    <img src="readme_fig/res_combination/res_combination.png" alt="Example of Residual Module" width="512"/>
</div>

**Description:**
pass

### test your train model

<div style="text-align:center;">
    <img src="readme_fig/forward_test.png" alt="test your train model" width="1024"/>
</div>

**Description:**
pass
