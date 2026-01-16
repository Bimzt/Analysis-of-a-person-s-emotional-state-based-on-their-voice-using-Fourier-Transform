# Analysis-of-a-person-s-emotional-state-based-on-their-voice-using-Fourier-Transform

1. Description

This Jupyter Notebook code is designed to process digital audio signals, extract relevant features, and train a Machine Learning model (Artificial Neural Network / MLP) to classify audio samples (e.g., Speech Emotion Recognition). The pipeline includes waveform visualization, MFCC (Mel-frequency Cepstral Coefficients) feature extraction, and model evaluation.

- IMPORTANT NOTE (DATASET)
  
Due to upload size limitations on GitHub or storage platforms, the dataset might be split.

Special Instruction: Please move all audio files currently located in the folder named audio 2 into the main folder named Audio. The script is configured to read data specifically from the Audio directory. Failing to merge these files may result in missing data or errors.

2. Prerequisites

Ensure you have the required Python libraries installed before running the notebook. You can install them via terminal or by running the installation cell in the notebook:

pip install librosa numpy pandas matplotlib seaborn scikit-learn

- Libraries Used:

librosa: For audio analysis and feature extraction.

matplotlib & seaborn: For data visualization (waveforms, heatmaps).

sklearn: For building the MLP model and evaluation metrics.

numpy & pandas: For numerical data handling.

3. Usage Steps

Step 1: Import Libraries
Run the first cell to import all necessary packages.

Step 2: Load & Visualize Audio
The code loads a sample audio file (e.g., '03-01-01-01-01-01-01.wav').

Output: An interactive audio player and a waveform plot of the raw audio signal will be displayed.

The code then performs Normalization to scale the audio amplitude. A second waveform showing the normalized signal is plotted.

Step 3: Feature Extraction (MFCC)
The system iterates through the files in the Audio folder.

It computes the MFCC (Mel-frequency Cepstral Coefficients) for each audio file. These are numerical features representing the timbre of the voice.

Step 4: Model Training
The dataset is split into Training and Testing sets using train_test_split.

An MLPClassifier (Multi-Layer Perceptron) neural network is trained on the training data.

Step 5: Evaluation & Visualization
The trained model predicts labels for the test set.

Accuracy: The classification accuracy score is printed.

Confusion Matrix: A heatmap is generated using seaborn to visualize the model's performance. It shows true labels versus predicted labels. Darker colors on the diagonal indicate correct predictions.
