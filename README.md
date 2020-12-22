# Neural network training examples
Artificial neural network examples for Python. These examples use the neural network package https://pypi.org/project/neuralnetwork/

Each example includes a trained network. You can either retrain the neural network or load the existing trained neural network from file

To run these examples you first must install the neural network package.

## Installation
```bash
$  pip install neuralnetwork
```

## Logging
### To log training to the file ./training.log create ./.env file with following parameters
```py
LOG_TRAINING=true
LOG_LEVEL=1
```
#### LOG_LEVEL=2 logs more details including network outputs and network error after each training example is presented.


