
# Sentiment Analysis Using Neural Networks on IMDB Dataset

## Overview
This project involves performing sentiment analysis on the IMDB dataset using various neural network techniques. The objective is to classify movie reviews as positive or negative based on the textual content. We employ several neural network architectures including simple neural networks, convolutional neural networks (CNN), and recurrent neural networks (RNN).

## Components Used
- **Python**: Programming language used for implementation.
- **Neural Networks**: Implemented using libraries like TensorFlow or PyTorch.
- **IMDB Dataset**: Movie reviews dataset used for sentiment analysis.

## Neural Network Techniques
1. **Simple Neural Network**: A basic feedforward neural network.
2. **Convolutional Neural Network (CNN)**: Used for extracting local features from text data.
3. **Recurrent Neural Network (RNN)**: Suitable for sequential data processing.

## Dataset
You can use a dataset of your choice for sentiment analysis. For this project, we used the IMDB dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets).

## Software Setup
### Prerequisites
- **Python**: Ensure you have Python installed. You can download it from [here](https://www.python.org/downloads/).
- **Jupyter Notebook**: Recommended for running and testing the code. Install it using `pip install notebook`.

### Libraries
- **TensorFlow/PyTorch**: For building and training neural networks.
- **NumPy**: For numerical operations.
- **Pandas**: For data manipulation.
- **Scikit-learn**: For preprocessing and evaluating models.

### Installation
1. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
2. Install the required libraries:
    ```bash
    pip install tensorflow numpy pandas scikit-learn jupyter
    ```

## Project Structure
- **`/src`**: Contains the source code for different neural network models.
- **`/data`**: Directory to store the dataset.
- **`/notebooks`**: Jupyter notebooks for experiments and analysis.
- **`/images`**: Images and visualizations generated during the project.

## Data Preparation
1. Download the IMDB dataset from [Kaggle](https://www.kaggle.com/datasets).
2. Place the dataset in the `/data` directory.

## Usage
### Running the Jupyter Notebooks
1. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2. Open and run the notebooks in the `/notebooks` directory to preprocess the data, train the models, and evaluate performance.

### Training and Evaluation
1. **Simple Neural Network**: Train a basic feedforward network on the IMDB dataset.
2. **CNN**: Train a convolutional neural network to capture local features in text data.
3. **RNN**: Train a recurrent neural network to process sequential data effectively.

## Acknowledgments
- **TensorFlow/PyTorch**: Libraries for building neural networks.
- **Kaggle**: For providing the IMDB dataset.
- **Python Community**: For the support and resources.
```
