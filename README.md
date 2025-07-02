# AI Detection and Humanization Tool

## Overview

The **AI Detection and Humanization Tool** is a Python-based application designed to detect AI-generated text, analyze its properties, and transform it into human-like writing. Built with advanced natural language processing (NLP) and machine learning, it combines rule-based detection, transformer models, and a custom artificial neural network (ANN) for accurate detection and text processing. The tool features a modern, user-friendly Tkinter GUI with a clean white theme, supporting functionalities like AI content detection, text humanization, summarization, and paraphrasing. It processes text inputs and files (`.txt`, `.docx`, `.pdf`) efficiently, making it ideal for researchers, writers, educators, and developers.

The project is hosted on GitHub: [Shadowinfinitywarrior/Ai-detection_Humanizer](https://github.com/Shadowinfinitywarrior/Ai-detection_Humanizer)

## Owner

- **Name**: Nithish K
- **Contact**: [nithishkathiravan123@gmail.com](mailto:nithishkathiravan123@gmail.com)

## Features

- **AI Content Detection**: Identifies AI-generated text and estimates probabilities for specific models (e.g., GPT-4, GPT-3, Claude, DeepSeek).
- **Text Humanization**: Converts AI-generated text to human-like writing using paraphrasing, synonym substitution, contractions, and stylistic variations.
- **Text Summarization**: Generates concise summaries using the BART model.
- **Text Paraphrasing**: Rephrases text with adjustable diversity using an ensemble of Parrot, T5, and BART models.
- **File Support**: Processes `.txt`, `.docx`, and `.pdf` files via drag-and-drop or file selection.
- **Training Capability**: Trains the ANN on human-written documents to improve humanization.
- **Responsive GUI**: Features tabs for input, results, processing, settings, learning, and history with a modern white theme.
- **Asynchronous Processing**: Handles intensive tasks in the background to ensure UI responsiveness.
- **Activity Log and History**: Tracks operations and maintains a history of actions.

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **System**: A CUDA-enabled GPU is recommended for optimal performance; CPU is supported.
- **Dependencies**: Listed in `requirements.txt`.

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Shadowinfinitywarrior/Ai-detection_Humanizer.git
   cd Ai-detection_Humanizer
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages:
   ```
   nltk
   tkinterdnd2
   python-docx
   pdfplumber
   transformers
   torch
   sentence-transformers
   textblob
   numpy
   python-dotenv
   ```

4. **Set Up Hugging Face Token**:
   - Obtain an API token from [Hugging Face](https://huggingface.co/settings/tokens).
   - Create a `.env` file in the project root:
     ```
     HF_TOKEN=your_hugging_face_token
     ```
   - The application loads the token using `python-dotenv`. Ensure `.env` is added to `.gitignore`.

5. **Run the Application**:
   ```bash
   python main.py
   ```

6. **NLTK Resources**:
   The application automatically downloads required NLTK resources (`punkt`, `stopwords`, `averaged_perceptron_tagger`, `wordnet`) on first run.

## Usage

1. **Launch the Application**:
   Run `main.py` to open the GUI.

2. **Input Tab**:
   - Enter text or drag-and-drop `.txt`, `.docx`, or `.pdf` files.
   - Use the "Load Files" button to select files.
   - Clear input with the "Clear" button.

3. **Results Tab**:
   - Displays AI detection probabilities for GPT-4, GPT-3, Claude, and DeepSeek, plus an overall AI probability score.
   - Lists suggestions to make text more human-like.

4. **Process Tab**:
   - **Analyze**: Detects AI patterns and updates the Results tab.
   - **Humanize**: Transforms text to sound human-like, shown in a new window.
   - **Summarize**: Generates a summary using the BART model.
   - **Paraphrase**: Rephrases text with adjustable diversity.

5. **Settings Tab**:
   - **Paraphrasing Diversity** (0.1–0.9): Controls how much paraphrased text differs from the original.
   - **Humanization Level** (0.1–1.0): Adjusts the intensity of humanization.

6. **Learn Tab**:
   - Upload `.docx` or `.pdf` files to train the ANN for better humanization.
   - View training status after processing.

7. **History Tab**:
   - Shows recent actions (e.g., file loading, analysis, humanization).
   - Click entries to view full content.

8. **Activity Log**:
   - Displays real-time operation logs at the bottom of the GUI.

## Project Structure

- **module.py**: Initializes NLTK resources.
- **main.py**: Core application logic, including:
  - `NLTKManager`: Handles NLTK resource setup.
  - `AIModelPatterns`: Defines AI model-specific detection patterns.
  - `ANN`: Custom neural network for text embedding learning.
  - `AdvancedParaphraser`: Manages paraphrasing with transformer models and ANN.
  - `AIContentDetector`: Performs AI detection, humanization, and summarization.
  - `AsyncTaskManager`: Runs background tasks for UI responsiveness.
  - `AIDetectorGUI`: Implements the Tkinter GUI with drag-and-drop support.

## Technical Details

### AI Detection
- **Rule-Based**: Uses regex for AI-specific phrases, passive voice, nominalizations, and hedging.
- **Pretrained Models**: Employs `roberta-base-openai-detector` for AI detection and `gpt2` for perplexity scoring.
- **Stylometric Features**: Analyzes sentence uniformity, lexical diversity, and n-gram repetition.
- **Model Detection**: Identifies signatures of GPT-4, GPT-3, Claude, and DeepSeek.

### Humanization
- **Paraphrasing**: Combines Parrot, T5, and BART models for diverse rephrasing.
- **Stylistic Enhancements**: Applies contractions, interjections, synonyms, and sentence variations.
- **ANN Training**: Fine-tunes embeddings on human-written texts.

### Summarization
- Uses `facebook/bart-large-cnn` for concise summaries.

### Asynchronous Processing
- Background threads handle file processing, analysis, and model tasks with progress bar feedback.

## Security Notes
- **Hugging Face Token**: Store your token in a `.env` file, not in source code, to prevent accidental exposure.
- **GitHub Push Protection**: Ensure no sensitive data (e.g., API tokens) is committed. Use `.gitignore` for `.env` files.

## Limitations
- Requires internet for initial model downloads.
- Performance may be slower without a GPU.
- Supports only `.txt`, `.docx`, and `.pdf` files.
- Model loading errors may fall back to rule-based detection, reducing accuracy.

## Future Enhancements
- Support additional file formats (e.g., `.rtf`, `.odt`).
- Improve humanization with context-aware transformations.
- Optimize model loading and caching.
- Enhance training with larger datasets.
- Add export options for processed text.

## Contributing
1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add your feature"`.
4. Push: `git push origin feature/your-feature`.
5. Open a pull request.

## License
MIT License. See [LICENSE](LICENSE) for details.

## Contact
For inquiries or issues, contact:
- **Nithish K**: [nithishkathiravan123@gmail.com](mailto:nithishkathiravan123@gmail.com)

*Last Updated: July 2, 2025*