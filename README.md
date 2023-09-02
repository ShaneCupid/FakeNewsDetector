# Fake News Detector
by: Shane C. 
## Overview
This project aims to detect fake news articles using machine learning techniques. The project consists of two main parts—Jupyter Notebooks for data processing and model training, and a Streamlit app for a front-end user interface.

## Technologies Used
- [Python 3](https://www.python.org/downloads/)
- [Jupyter Notebook](https://jupyter.org/install)
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Scikit-Learn](https://scikit-learn.org/stable/)

## Description
The primary goal of this project is to classify news articles as either "fake" or "real" based on their content. The Jupyter Notebooks handle data cleaning, preprocessing, and machine learning, while the Streamlit app provides a user interface for real-time fake news detection.

### Jupyter Notebooks
#### Data_Preparation.ipynb
- **Input**: Raw news articles in CSV format.
- **Output**: Cleaned and preprocessed data in CSV format.
- **Description**: This notebook prepares the data for machine learning by cleaning and transforming text data into a suitable format.

#### Fake_News_Detection.ipynb
- **Input**: Cleaned data from `fake.ipynb`.
- **Output**: Machine learning model trained to detect fake news.
- **Description**: This notebook applies machine learning algorithms to classify news articles. It uses TF-IDF for feature extraction and Linear SVC for the classification.

### Streamlit App
- **File**: `app.py`
- **Description**: This Streamlit app allows the user to paste a news article for real-time classification as "fake" or "real".

## How It Works

### Learning from Data
The model begins its operation by learning from a pre-labeled dataset, where each article is already classified as either "REAL" or "FAKE." By training on this data, the model learns to recognize the patterns and characteristics that differentiate fake articles from real ones.

### Feature Engineering
The Text Frequency-Inverse Document Frequency (TF-IDF) technique is used to transform each article into a numerical form. This numerical representation captures the importance of each word in the context of the entire dataset, facilitating the machine learning process.

### Training the Model
Using the numerical data, the model is trained to understand the patterns that separate fake news from real news. In technical terms, the model attempts to find a hyperplane in a multi-dimensional space that acts as a "boundary" between fake and real articles. 

### Testing and Prediction
Once the model is trained, it can then be tested on new, unlabeled data. When given a new article (which is also converted into numerical form via TF-IDF), the model places it on one side of the boundary or the other. Based on this placement, the article is labeled as either "fake" or "real."



## Setup/Installation Requirements
1. Install Python 3 from the [official website](https://www.python.org/downloads/).
2. Install Jupyter Notebook, preferably via Anaconda. Here's a [guide](https://www.datacamp.com/community/tutorials/installing-jupyter-notebook).
3. Install Streamlit by running `pip install streamlit` in your command line.
4. Install the required Python libraries by running `pip install pandas numpy scikit-learn` in your command line.
5. Clone this repository to your local machine.
6. Navigate to the local repository and run `jupyter notebook` to open the Jupyter Notebook in your browser.
7. To run the Streamlit app, navigate to its directory and run `streamlit run app.py`.

## Contact Information
For any queries, please feel free to contact me at [your contact information].

## License
This project is licensed under the [MIT License](LICENSE).
