# Real-Time Sign Language Recognition System Using Convolutional Neural Networks

A deep learning-based system for recognizing American Sign Language (ASL) alphabet gestures in real-time, achieving 99.19% validation accuracy.

## View Full Project

**Note:** The full notebook with outputs is too large for GitHub's viewer.

**View Interactive Notebook with All Results:**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NniTXEwQVO_z_qxviSVPXeosLFFqV43N?usp=sharing)

*Click above to view the complete project with training results, visualizations, and real-time recognition demo.*

## Overview

This project implements a custom Convolutional Neural Network to translate ASL fingerspelling into text. The system captures gestures via webcam, recognizes individual letters, and forms complete words and sentences - making it a practical tool for accessibility and communication.

## Features

- **Real-time Recognition**: Webcam-based gesture capture and instant prediction
- **Word Formation**: Automatically collects letters to build words
- **Sentence Building**: Space gesture adds completed words to sentences
- **High Accuracy**: 99.19% validation accuracy on 87,000+ images
- **29 Classes**: Recognizes A-Z letters plus space, delete, and nothing gestures

## Technical Details

### Model Architecture
- Custom CNN with 4 convolutional blocks
- 8.7M parameters
- Architecture: Conv layers (32→64→128→256 filters) + 2 FC layers
- Dropout (0.5) for regularization

### Training Configuration
- **Framework**: PyTorch
- **Dataset**: ASL Alphabet Dataset (87,000 images, 29 classes)
- **Hardware**: Tesla T4 GPU (Google Colab)
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 5
- **Final Accuracy**: 99.19% validation, 97.76% training

## Dataset

**Source**: [ASL American Sign Language Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)

- Total Images: 223,074
- Training: 178,459
- Validation: 44,615
- Classes: 29 (A-Z + space + del + nothing)

## Installation & Setup

### Requirements
```bash
pip install torch torchvision opencv-python pillow matplotlib seaborn numpy
```

### Running on Google Colab
1. Open the notebook in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells sequentially
4. Follow webcam capture instructions for real-time recognition

## Usage

### Training the Model
The notebook includes complete training pipeline. Simply run all cells to:
1. Download and prepare the dataset
2. Build the CNN architecture
3. Train for 5 epochs (~30 minutes on T4 GPU)
4. Evaluate and visualize results

### Real-Time Recognition
```python
# Press Enter to capture sign
# Make ASL gesture
# Click "Capture Sign" button
# System displays: Current Letter | Current Word | Sentence
```

**Controls:**
- Press Enter: Capture new sign
- Sign 'space': Add word to sentence
- Sign 'del': Delete last letter
- Type 'done': Finish session
- Type 'reset': Clear all text

## Results

**Model Performance:**
- Training Accuracy: 97.76%
- Validation Accuracy: 99.19%
- Training Loss: 0.0703

**Per-Class Performance:**
- Average Precision: 0.99
- Average Recall: 0.99
- F1-Score: 0.99

See confusion matrix and detailed classification report in the notebook.

## Project Structure

```
├── notebook.ipynb          # Main Jupyter notebook
├── sign_language_model.pth # Trained model weights
├── README.md               # This file
└── results/                # Training visualizations
    ├── accuracy_plot.png
    ├── confusion_matrix.png
    └── sample_predictions.png
```

## Applications

- **Accessibility**: Real-time communication tool for deaf/hard-of-hearing individuals
- **Education**: Interactive platform for learning ASL alphabet
- **Assistive Technology**: Foundation for more comprehensive sign language translation systems

## Future Enhancements

- Continuous video stream recognition (replace frame-by-frame capture)
- Full ASL word vocabulary beyond fingerspelling
- Multi-hand gesture support
- Mobile app deployment
- Integration with text-to-speech for audio output

## Technologies Used

- **Deep Learning**: PyTorch, torchvision
- **Computer Vision**: OpenCV, PIL
- **Data Processing**: NumPy, pandas
- **Visualization**: Matplotlib, Seaborn
- **Development**: Google Colab, Jupyter Notebook

## License

This project is available for educational and research purposes.

## Acknowledgments

- Dataset: Debashish Sau (Kaggle)
- Training Infrastructure: Google Colab

## Author

Snigdha Bairi

---

**Note**: This project focuses on ASL fingerspelling (letter-by-letter). Full ASL includes thousands of word-level signs not covered in this implementation.
