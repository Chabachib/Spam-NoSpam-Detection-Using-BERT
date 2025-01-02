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
├── models/
│   ├── bert_classifier.pth  # Trained BERT model
├── notebooks/
│   ├── notebook.ipynb       # The project's notebook
├── translated_models/
│   ├── opus-mt-fr-en/       # French to English translation model
│   ├── opus-mt-es-en/       # Spanish to English translation model
│   └── opus-mt-ar-en/       # Arabic to English translation model
├── requirements.txt         # Project dependencies
```

## 📖 Project Overview
### Libraries Used
PyTorch, Transformers (Provides the pre-trained BERT model and tokenizer), Scikit-learn, Pandas, Matplotlib/Seaborn, WordCloud

### Dataset Preparation
We used datasets sourced from Kaggle, here's the [Link to Dataset](https://kaggle.com/datasets/574375de5edc46f705ed8fbcd63d72430f601e22adee7fda2bc94a69ee7160b5), which consisted of two files: messages.csv (email dataset) and spam.csv (SMS dataset). 
Both datasets were loaded into pandas DataFrames using `pd.read_csv()` with the `on_bad_lines='skip'` parameter to handle any malformed rows. Unnecessary columns were removed from each dataset to focus only on the text and corresponding labels, and the target labels were standardized to ensure consistency by mapping `0` to `"ham"` (not spam) and `1` to `"spam"`. The column order was then rearranged for uniformity, ensuring both datasets contained a text column for the message content and a label column for classification. 
Finally, the cleaned and aligned DataFrames were concatenated into a single dataset for further analysis and modeling.


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
