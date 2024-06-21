# app.py
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
from feature_extraction import load_dataset, train_classifiers, extract_features, allowed_file
import matplotlib.pyplot as plt
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

app = Flask(__name__)
app.debug = True
# Define the file upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load dataset and train classifiers
X_train, X_test, y_train, y_test = load_dataset()
knn_classifier, svm_classifier, rf_classifier = train_classifiers(X_train, y_train)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return "File Missing"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract features from the uploaded file
        features = extract_features(file_path)
        print("Extracted features:", features)

        # Predict the class label using each classifier
        knn_pred = knn_classifier.predict([features])[0]
        svm_pred = svm_classifier.predict([features])[0]
        rf_pred = rf_classifier.predict([features])[0]

        return f"KNN Prediction: {knn_pred} | SVM Prediction: {svm_pred} | Random Forest Prediction: {rf_pred}"

    else:
        return "Only .wav files are accepted."


@app.route('/metrics', methods=['GET'])
def metrics():
    # Predict labels for test set
    knn_pred = knn_classifier.predict(X_test)
    svm_pred = svm_classifier.predict(X_test)
    rf_pred = rf_classifier.predict(X_test)

    # Calculate metrics
    knn_accuracy = accuracy_score(y_test, knn_pred)
    knn_precision = precision_score(y_test, knn_pred, average='weighted')
    knn_recall = recall_score(y_test, knn_pred, average='weighted')
    knn_f1 = f1_score(y_test, knn_pred, average='weighted')

    svm_accuracy = accuracy_score(y_test, svm_pred)
    svm_precision = precision_score(y_test, svm_pred, average='weighted')
    svm_recall = recall_score(y_test, svm_pred, average='weighted')
    svm_f1 = f1_score(y_test, svm_pred, average='weighted')

    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred, average='weighted')
    rf_recall = recall_score(y_test, rf_pred, average='weighted')
    rf_f1 = f1_score(y_test, rf_pred, average='weighted')

    # Plot bar chart
    classifiers = ['KNN', 'SVM', 'Random Forest']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    knn_metrics = [knn_accuracy * 100, knn_precision * 100, knn_recall * 100, knn_f1 * 100]
    svm_metrics = [svm_accuracy * 100, svm_precision * 100, svm_recall * 100, svm_f1 * 100]
    rf_metrics = [rf_accuracy * 100, rf_precision * 100, rf_recall * 100, rf_f1 * 100]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, knn_metrics, width, label='KNN')
    rects2 = ax.bar(x, svm_metrics, width, label='SVM')
    rects3 = ax.bar(x + width, rf_metrics, width, label='Random Forest')

    ax.set_ylabel('Scores (%)')
    ax.set_title('Classification Metrics by Classifier')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    # Set y-axis limits to ensure it goes up to 100%
    plt.ylim(0, 100)
    # Add labels to the bars
    add_labels(ax, rects1)
    add_labels(ax, rects2)
    add_labels(ax, rects3)
    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Return the plot as a PNG file
    return send_file(img, mimetype='image/png')

def add_labels(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')


if __name__ == '__main__':
    app.run(debug=True)
