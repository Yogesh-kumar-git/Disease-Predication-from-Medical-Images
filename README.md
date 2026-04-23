# 🩺 Disease Prediction from Medical Images

A deep learning-based web application that predicts diseases from medical images using Convolutional Neural Networks (CNNs). The system currently supports **Pneumonia detection** from chest X-rays  and Brain Tumour Detection from MRI Sacans

---

## 📌 Project Overview

| Module | Task | Classes | Input Size |
|---|---|---|---|
| Chest X-Ray Analysis | Pneumonia Detection | NORMAL / PNEUMONIA | 128 × 128 |
| Brain Tumour Detection | Tumour Classification | Negative / Positive | 224 × 224 |

---

## 🚀 Features

- 🖼️ Upload chest X-ray or brain MRI scan directly from the browser
- 🤖 Instant AI-powered prediction using trained deep learning models
- 📊 Confidence score displayed alongside the prediction
- 🔄 Two separate trained models — one per disease module
- 🧠 Transfer learning with **VGG16** (X-Ray) and **MobileNetV2** (Brain Tumour)
- 🌐 Clean and minimal Streamlit web interface


## 🧠 Model Details

### 1. Chest X-Ray — Pneumonia Detection

- **Dataset:** Chest X-Ray Images (Pneumonia) — Kaggle
- **Classes:** `NORMAL`, `PNEUMONIA`
- **Architectures Trained:**
  - Custom CNN — 4 convolutional blocks (32 → 64 → 128 → 256 filters), BatchNormalization, Dropout
  - **VGG16** (Transfer Learning, frozen base) — fine-tuned Dense head
- **Training Strategy:** Data augmentation (rotation, shift, zoom, horizontal flip), EarlyStopping, ReduceLROnPlateau
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam

### 2. Brain Tumour Detection

- **Dataset:** Brain Tumor Dataset — Kaggle (`praneet0327/brain-tumor-dataset`)
- **Classes:** `Negative` (No Tumour), `Positive` (Tumour Present)
- **Architectures Trained:**
  - Custom CNN — 3 convolutional blocks (128 → 128 → 256 filters), BatchNormalization, L2 Regularization, Dropout
  - **MobileNetV2** (Transfer Learning, frozen base) — GlobalAveragePooling + Dense head
- **Training Strategy:** Data augmentation, EarlyStopping, ModelCheckpoint (resumes training from checkpoint)
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam

---

## 🌐 Streamlit Web App

The Streamlit app allows users to:

1. Choose the scan type — **Chest X-Ray** or **Brain MRI**
2. Upload their scan image (`.jpg`, `.jpeg`, `.png`)
3. View the prediction result — e.g., *"PNEUMONIA Detected"* or *"No Tumour Found"*
4. See the model confidence score

### Running the App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Disease-Prediction-from-Medical-Images.git
cd Disease-Prediction-from-Medical-Images
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Trained Models

Place your saved `.h5` model files inside the `models/` directory:

```
models/chest_xray_model.h5
models/brain_tumour_model.h5
```

> If you want to retrain, run the respective Jupyter notebooks.

### 5. Launch the App

```bash
streamlit run app.py
```

---

## 📦 Requirements

```
tensorflow>=2.10.0
streamlit
numpy
opencv-python
matplotlib
seaborn
scikit-learn
Pillow
kagglehub
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 📊 Training Results

### Chest X-Ray (VGG16)
- Training with data augmentation and early stopping over 10 epochs
- Evaluated on held-out test set with accuracy and loss curves

### Brain Tumour (MobileNetV2)
- Initial training: 12 epochs with EarlyStopping
- Resumed training: additional 5 epochs from checkpoint (total 17 epochs)
- Final evaluation using classification report and confusion matrix

---

## 🖥️ How to Use the Web App

1. Open the app at `http://localhost:8501`
2. Select the **Scan Type** from the sidebar or dropdown:
   - 🫁 Chest X-Ray
   - 🧠 Brain MRI Scan
3. Click **Upload Image** and select your scan file
4. The model will process the image and display:
   - Prediction label
   - Confidence percentage

> ⚠️ **Disclaimer:** This tool is for educational and research purposes only. It is not a substitute for professional medical diagnosis.

---

## 🔬 Datasets Used

| Dataset | Source |
|---|---|
| Chest X-Ray Images (Pneumonia) | [Kaggle — Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| Brain Tumor Dataset | [Kaggle — praneet0327](https://www.kaggle.com/datasets/praneet0327/brain-tumor-dataset) |

---

## 👨‍💻 Author

**Yogesh Kumar**
- LinkedIn: [linkedin.com/in/your-profile](https://www.linkedin.com/in/yogesh-kumar-362324298)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgements

- TensorFlow / Keras for deep learning framework
- Streamlit for the web application framework
- Kaggle for providing open-source medical imaging datasets
- VGG16 and MobileNetV2 pre-trained weights from ImageNet
