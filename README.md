# Multilingual Spam Detection with BERT

A powerful spam detection system that combines BERT classification with multilingual support. The system can detect spam across multiple languages by first translating the input text to English and then applying a highly accurate BERT-based classifier trained on spam/ham emails and SMS messages.

## 🎯 Features

- 99% accurate BERT-based spam classification
- Multi-language support (English, French, Spanish, Arabic)
- Real-time translation to English before classification
- Interactive Streamlit dashboard
- Pre-trained translation models using Opus-MT

## 📁 Project Structure

```
spam-detection/
├── app/
│   ├── main.py              # Streamlit dashboard implementation
│   ├── utils.py             # Utility functions for model loading and translation
│   ├── classifier.py        # BERT Classifier and predict spam function
├── models/
│   ├── bert_classifier.pth  # Trained BERT model
├── notebooks/
│   ├── notebook.ipynb       # The project's notebook
├── translated_models/
│   ├── opus-mt-fr-en/       # French to English translation model
│   ├── opus-mt-es-en/       # Spanish to English translation model
│   └── opus-mt-ar-en/       # Arabic to English translation model
├── requirements.txt         # Project dependencies
├── README.md                # README file for clarifications
```

## 📖 Project Overview
### 1. Libraries Used
PyTorch, Transformers (Provides the pre-trained BERT model and tokenizer), Scikit-learn, Pandas, Matplotlib/Seaborn, WordCloud

### 2. Dataset Preparation
We used datasets sourced from Kaggle, here's the [Link to Dataset](https://kaggle.com/datasets/574375de5edc46f705ed8fbcd63d72430f601e22adee7fda2bc94a69ee7160b5), which consisted of two files: messages.csv (email dataset) and spam.csv (SMS dataset). 
Both datasets were loaded into pandas DataFrames using `pd.read_csv()` with the `on_bad_lines='skip'` parameter to handle any malformed rows. Unnecessary columns were removed from each dataset to focus only on the text and corresponding labels, and the target labels were standardized to ensure consistency by mapping `0` to `"ham"` (not spam) and `1` to `"spam"`. The column order was then rearranged for uniformity, ensuring both datasets contained a text column for the message content and a label column for classification. 
Finally, the cleaned and aligned DataFrames were concatenated into a single dataset for further analysis and modeling.

### 3. Model Architecture

The BERTClassifier leverages the powerful natural language understanding capabilities of the pre-trained BERT model (bert-base-uncased) provided by Hugging Face. It is specifically tailored for the binary classification task of identifying spam versus ham messages. The architecture is designed to balance robustness, accuracy, and simplicity, consisting of the following key components:

- **<ins>BERT Backbone:<ins>**
The pre-trained BERT model serves as the foundation of the classifier. It is responsible for generating contextual embeddings for the input text. These embeddings encapsulate the semantic and syntactic nuances of the text, making the model highly effective at understanding complex language patterns.

- **<ins>Dropout Layer:<ins>**
A dropout layer is added after the BERT output to mitigate overfitting. This layer randomly zeroes out a fraction of the neurons during training, ensuring the model generalizes well to unseen data.

- **<ins>Fully Connected Layer:<ins>**
The classifier head consists of a fully connected layer that takes the embeddings from the BERT model and maps them to two output classes—spam and ham. This layer translates the high-dimensional representations from BERT into a simple decision boundary for binary classification.

- **<ins>Cross-Entropy Loss Function:<ins>**
The model uses the cross-entropy loss function to calculate the discrepancy between the predicted and true class labels. This loss guides the optimization process during training.

- **<ins>Optimization and Scheduling:<ins>**
The AdamW optimizer, known for its adaptive learning rates and weight decay properties, is used to update the model weights. A linear learning rate scheduler with warm-up steps ensures smooth optimization, preventing abrupt changes that might destabilize training.

### 4. Text Prediction
The `predict_sentiment` function enables the model to make predictions on raw input text, determining whether it is classified as `"spam"` or `"ham"`. The function follows a structured process to ensure accurate predictions:

- **<ins>Text Encoding:<ins>** The raw text input is tokenized using the same tokenizer used during model training. This step transforms the text into a numerical format that the model can process. The tokenizer generates `input_ids` and `attention_mask` with consistent padding and truncation to maintain the required input size.

- **<ins>Model Inference:<ins>** The encoded text is passed through the trained BERTClassifier, which processes the input and generates a set of logits (unnormalized probabilities) for each class.

- **<ins>Class Mapping:<ins>** The predicted logits are converted to class indices using torch.max, and the class index is mapped to its corresponding label: `"spam"` for index `1` and `"ham"` for index `0`.

### 5. Parameters
- `bert_model_name:` The pre-trained BERT model to use (bert-base-uncased).
- `num_classes:` Number of output classes (2: spam and ham).
- `max_length:` Maximum length for truncating/padding text inputs (128 tokens).
- `batch_size:` Batch size for training and evaluation (16).
- `num_epochs:` Number of training epochs (4).
- `learning_rate:` Learning rate for the optimizer (2e-5).

### 6. Workflow
- The dataset is split into training and validation sets using train_test_split.
- SpamClassificationDataset converts text and labels into a format compatible with BERT.
- The BERTClassifier is initialized and moved to the appropriate device (CPU or GPU).
- The model is trained for 4 epochs.
- After each epoch, the validation accuracy and classification report are printed.
- The classification report includes precision, recall, and F1-score for both classes.

### 7. Results
Achieved validation accuracy above 99% after the first epoch, which remains consistent across all epochs.
Classification Report:
- `Class 0 (Ham):` Very high precision, recall, and F1-scores (close to 1.00).
- `Class 1 (Spam):` Slightly lower recall (e.g., 0.95–0.97) but overall strong performance.

### 8. Translation Models
For our translation models, we utilized the Open Neural Machine Translation [Opus-MT](https://github.com/Helsinki-NLP/Opus-MT) models, developed and maintained by the Helsinki-NLP team. These models are built using the Marian NMT framework, a highly efficient and scalable platform for neural machine translation. The Opus-MT models support a wide range of languages and have been fine-tuned on various multilingual corpora, making them highly versatile for translation tasks.

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Chabachib/Spam-NoSpam-Detection-Using-BERT.git
cd Spam-NoSpam-Detection-Using-BERT
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Start the Streamlit dashboard:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Enter your text in any supported language (English, French, Spanish, or Arabic)

4. The system will:
   - Detect the input language
   - Translate the text to English if necessary
   - Classify the text as spam or ham
   - Display the results

## 🛠️ Technical Details

### BERT Classifier
- Based on BERT base model
- Fine-tuned on spam/ham dataset
- Achieves 99% accuracy on test set
- Optimized for both email and SMS spam detection

### Translation Models
The project uses Opus-MT models for translation:
- French to English: `opus-mt-fr-en`
- Spanish to English: `opus-mt-es-en`
- Arabic to English: `opus-mt-ar-en`

## 📋 Requirements

Key dependencies include:
- Python 3.8+
- PyTorch
- Transformers
- Streamlit
- Sentencepiece
- See `requirements.txt` for complete list

## 🙏 Acknowledgments

- HuggingFace Transformers library
- Opus-MT for translation models
- BERT paper authors
- Streamlit team

## 📧 Contact

Project Link: [https://github.com/Chabachib/Multilingual-Spam-NoSpam-Detection-Using-BERT](https://github.com/Chabachib/Multilingual-Spam-NoSpam-Detection-Using-BERT)
