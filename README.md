# 🤖 Hand Gesture Recognition using Deep Learning (2D CNN, Bi-LSTM, CNN-LSTM)

This repository presents a deep learning-based hand gesture recognition system using the [LeapGestRecog Dataset](https://www.kaggle.com/datasets/kmader/leapgestrecog). The project compares the performance of three architectures:
- **2D CNN**: Focuses on spatial features from static gesture images.
- **Bi-LSTM**: Learns temporal dependencies in image sequences.
- **CNN-LSTM**: A hybrid model that combines spatial and temporal learning for enhanced recognition accuracy.

---

## 📁 Dataset

The [LeapGestRecog dataset](https://www.kaggle.com/datasets/kmader/leapgestrecog) consists of grayscale hand gesture images from 10 users performing 10 gesture types with over 200,000 frames.

---

## 🧠 Models Implemented

1. ### 2D CNN
   - Trained on individual grayscale frames.
   - Suitable for static gesture classification.
   - Lacks temporal context.

2. ### Bi-LSTM
   - Uses sequences of raw image frames (flattened or preprocessed).
   - Captures motion over time but lacks spatial feature richness.

3. ### CNN-LSTM (Best Performing Model)
   - Combines CNN for spatial feature extraction and LSTM for temporal sequence learning.
   - Achieved the highest accuracy among all models.

---

## 📊 Results Summary

| Model      | Training Accuracy | Validation Accuracy |
|------------|-------------------|---------------------|
| 2D CNN     | ~98.5%            | ~95.2%              |
| Bi-LSTM    | ~97.3%            | ~93.8%              |
| CNN-LSTM   | ~99.2%            | ~96.7%              |

- Confusion matrices show minimal misclassifications.
- High precision and recall across most gesture classes.
- Accurate real-time predictions on unseen samples.

---

## 🛠️ Dependencies

Make sure to install the following dependencies:
'pip install numpy matplotlib opencv-python tensorflow scikit-learn'

🧪 How to Run
Clone the repository:

git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition
Download and extract the dataset from Kaggle:
LeapGestRecog Dataset on Kaggle

Run the model scripts:

For 2D CNN: 2d_cnn_model.ipynb

For Bi-LSTM: bilstm_model.ipynb

For CNN-LSTM: cnn_lstm_model.ipynb

Evaluate results and visualize training accuracy, loss, confusion matrix, precision & recall.

📌 Project Highlights
  ✋ Real-time gesture prediction
  🧠 Spatiotemporal learning with CNN-LSTM
  📈 Visualized training curves and metrics
  ✅ Works on grayscale hand gesture images with high accuracy

📧 Contact
For any queries or collaborations, feel free to reach out at:
  Email: asmigit834@gmail.com
  GitHub: asmidagar

⭐ License
This project is open-source under the MIT License.








