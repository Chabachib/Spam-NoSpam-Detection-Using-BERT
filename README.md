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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace Transformers library
- Opus-MT for translation models
- BERT paper authors
- Streamlit team

## 📧 Contact

Project Link: [https://github.com/Chabachib/Multilingual-Spam-NoSpam-Detection-Using-BERT](https://github.com/Chabachib/Multilingual-Spam-NoSpam-Detection-Using-BERT)
