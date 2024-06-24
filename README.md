# ComfyUI Build and Train Your Network

This is an extension project aimed at customizing neural network layers (such as linear layers, convolutional layers,
etc.) in ComfyUI and providing simplified task training functionality. With this project, you can easily build custom
neural network layers in a graphical manner in ComfyUI and complete training, thereby extending its functionality.

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

   In ComfyUI, tensor calculations do not produce gradients. To resolve this issue, you need to modify
   the `ComfyUI/execution.py` file by commenting out `with torch.inference_mode():` and moving all the code within this
   context manager one level forward.

    ```bash
    copy .\replace\execution.py ..\..\execution.py
    ```

   **Note:** Modifying `ComfyUI/execution.py` might cause issues with image generation. If this happens, don't worry;
   you can restore the original state by running the following command:

    ```bash
    copy .\replace\execution1.py ..\..\execution.py
    ```

## Description

![Fully Connected Layer](readme_fig/linear/linear.png)

**Fully Connected Layer Description**

**Structure and Connections:**
In a fully connected layer, each neuron is connected to all neurons in the previous layer. If the previous layer has \( n \) neurons and the current layer has \( m \) neurons, then each neuron in the current layer will have \( n \) inputs. Therefore, the weight matrix \( W \) of the fully connected layer is an \( m \times n \) matrix, and the bias vector \( b \) is a vector of length \( m \).

**Mathematical Representation:**
Let the input vector be \( \mathbf{x} \), the weight matrix be \( \mathbf{W} \), and the bias vector be \( \mathbf{b} \). The output \( \mathbf{y} \) of the fully connected layer can be represented as:
\[ \mathbf{y} = \mathbf{W} \mathbf{x} + \mathbf{b} \]
Each element \( y_i \) of the output is obtained by calculating the dot product of the input vector and the corresponding row of the weight matrix, then adding the bias term.