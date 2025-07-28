# Model-Evaluation-and-Hyperparameter-Tuning-CSI-main

# 📊 Digit Classification with Hyperparameter Tuning

This project implements and compares different machine learning models — **Random Forest**, **Logistic Regression**, and **Support Vector Machine (SVM)** — on the Digits dataset from `scikit-learn`. The project demonstrates the impact of hyperparameter tuning using **GridSearchCV** and **RandomizedSearchCV**, along with evaluation using standard classification metrics.

## 🔍 Project Highlights

- Digit classification using the `load_digits()` dataset.
- Comparison of three classifiers: Random Forest, Logistic Regression, and SVM.
- Added Gaussian noise to increase the complexity of classification.
- Feature scaling using `StandardScaler`.
- Hyperparameter optimization using:
  - Grid Search (`GridSearchCV`)
  - Random Search (`RandomizedSearchCV`)
- Evaluation using Accuracy, Precision, Recall, and F1-Score.
- Confusion Matrix visualization for the best-performing model.

## 📁 Dataset

The project uses the built-in [`load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) dataset from `scikit-learn`, which consists of 8x8 grayscale images of handwritten digits (0 through 9).

- **Number of samples**: 1797
- **Features**: 64 (8x8 pixel values)
- **Classes**: 10 (digits 0 to 9)

Gaussian noise is added to the features to simulate real-world noise and increase complexity.

## 🧠 Models Used

- **Random Forest Classifier**
- **Logistic Regression**
- **Support Vector Machine (SVM)**

Each model is trained:
1. With default parameters
2. Using GridSearchCV
3. Using RandomizedSearchCV

## 🧪 Evaluation Metrics

- **Accuracy**
- **Precision (weighted)**
- **Recall (weighted)**
- **F1-Score (weighted)**

The **best model** is selected based on the **F1-Score**.

## 📊 Results

All models are evaluated on the test set (20% of the data), and performance is logged and compared. The confusion matrix for the best-performing model is visualized using Seaborn heatmap.

## 🧰 Requirements

Install required dependencies using pip:

bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
▶️ How to Run
bash
Copy
Edit
python digit_classification.py
Make sure your script file is named accordingly. This will:

Train all models

Perform Grid and Random search

Display metrics

Plot the confusion matrix of the best model

📈 Sample Output
yaml
Copy
Edit
Best Model: Random Forest (RandomizedSearchCV)
Accuracy: 0.955
Precision: 0.956
Recall: 0.955
F1-Score: 0.955
(Displayed by the script)

📌 Notes
A reduced training size is used intentionally to simulate performance on limited data.

You can tweak the amount of Gaussian noise added to see how models behave under noisy input.

All computations are done using CPU.

🧑‍💻 Author
Yuvraj Singh

📜 License
This project is licensed under the MIT License.

yaml
Copy
Edit

---

Let me know if you'd like to include:
- Jupyter Notebook version
- A `requirements.txt`
- Model saving with `joblib` or `pickle`
- GitHub badges or actions

I can generate those too.









