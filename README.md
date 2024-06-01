# Sentiment Analysis of Customer Reviews

## Overview

This project aims to develop a sentiment analysis model capable of classifying customer reviews into positive, negative, or neutral categories. By analyzing the sentiment of customer feedback, businesses can gain valuable insights into customer satisfaction levels and identify areas for improvement in products or services.

## Project Structure

- **data**: This directory contains the dataset used for training and testing the sentiment analysis model. Ensure that the dataset file (`customer_reviews.csv`) is located here.
  
- **src**: This directory contains the source code for the sentiment analysis model.
  - `sentiment_analysis.py`: Python script for preprocessing the text data, training the machine learning model, and evaluating its performance.
  
- **README.md**: This file provides an overview of the project, instructions for running the code, and details about the dataset.

## Setup Instructions

1. **Clone the Repository**: Begin by cloning this repository to your local machine.

    ```
    git clone https://github.com/Ken2ie/Sentiment-Analysis-of-Customer-Reviews.git
    ```

2. **Install Dependencies**: Ensure you have Python installed on your machine. Install the required dependencies using pip.

    ```
    pip install nltk scikit-learn pandas
    ```

3. **Download NLTK Resources**: Run the following commands in your Python environment to download the necessary NLTK resources.

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

4. **Prepare Dataset**: Place your dataset file (`customer_reviews.csv`) in the `data` directory. Ensure that the file contains two columns: `text` for customer reviews and `sentiment` for corresponding sentiment labels (e.g., 'positive', 'negative', 'neutral').

5. **Run the Code**: Navigate to the `src` directory and execute the `sentiment_analysis.py` script to preprocess the data, train the model, and evaluate its performance.

    ```
    cd src
    python sentiment_analysis.py
    ```

6. **View Results**: After running the script, you'll see the accuracy score and classification report indicating the model's performance in classifying customer reviews.

## Dataset Information

The dataset (`customer_reviews.csv`) consists of customer reviews along with their corresponding sentiment labels. It is used to train and test the sentiment analysis model. Feel free to replace this dataset with your own data if needed.

## Contact Information

For any inquiries or assistance regarding this project, please contact:

Pandit Alabi  
Email: panditmckenzie@gmail.com  
LinkedIn: [linkedin.com/in/pandit-alabi-238196201](https://www.linkedin.com/in/pandit-alabi-238196201)  
GitHub: [github.com/Ken2ie](https://github.com/Ken2ie)
