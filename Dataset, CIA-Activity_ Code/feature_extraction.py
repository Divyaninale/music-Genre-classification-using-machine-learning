import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import librosa
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import pickle
# Define global scaler variable
scaler = None

ALLOWED_EXTENSIONS = {'wav'}
print("feature_extraction module imported")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define selected_features here
selected_features = [
    'chroma_stft_mean', 'chroma_stft_var',
    'rms_mean', 'rms_var',
    'spectral_centroid_mean', 'spectral_centroid_var',
    'spectral_bandwidth_mean', 'spectral_bandwidth_var',
    'rolloff_mean', 'rolloff_var',
    'zero_crossing_rate_mean', 'zero_crossing_rate_var',
    'harmony_mean', 'harmony_var',
    'perceptr_mean', 'perceptr_var',
    'tempo',
    'mfcc1_mean', 'mfcc1_var',
    'mfcc2_mean', 'mfcc2_var',
    'mfcc3_mean', 'mfcc3_var',
    'mfcc4_mean', 'mfcc4_var',
    'mfcc5_mean', 'mfcc5_var',
    'mfcc6_mean', 'mfcc6_var',
    'mfcc7_mean', 'mfcc7_var',
    'mfcc8_mean', 'mfcc8_var',
    'mfcc9_mean', 'mfcc9_var',
    'mfcc10_mean', 'mfcc10_var',
    'mfcc11_mean', 'mfcc11_var',
    'mfcc12_mean', 'mfcc12_var',
    'mfcc13_mean', 'mfcc13_var',
    'mfcc14_mean', 'mfcc14_var',
    'mfcc15_mean', 'mfcc15_var',
    'mfcc16_mean', 'mfcc16_var',
    'mfcc17_mean', 'mfcc17_var',
    'mfcc18_mean', 'mfcc18_var',
    'mfcc19_mean', 'mfcc19_var',
    'mfcc20_mean', 'mfcc20_var',
    'label'
]

def load_dataset():
    # Load the dataset and preprocess
    print("Data set loading in process...")
    dataset = pd.read_csv("data.csv")

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    dataset[selected_features[:-1]] = imputer.fit_transform(dataset[selected_features[:-1]])
    
    global scaler  # Declare scaler as global variable
    
    # Handling inconsistency and outliers with normalization
    scaler = StandardScaler()
    dataset[selected_features[:-1]] = scaler.fit_transform(dataset[selected_features[:-1]])

    # Save the scaler object using pickle
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Selecting necessary features
    dataset = dataset[selected_features]  # Moved selected_features here
    
    # Split dataset into features (X) and labels (y)
    X = dataset.drop(columns=["label"])  # Exclude label column
    y = dataset["label"]
    print(X.shape)

    # Split the dataset into training and testing sets with shuffling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=50)
    print("Data set split successfully!")
    return X_train, X_test, y_train, y_test 

def preprocess_data(X_train, X_test, y_train):
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=20)  # Choose the number of components
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    return X_train_pca, X_test_pca

def train_classifiers(X_train, y_train):
    print("Training classifiers...")
    
    # Train KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=4)
    knn_classifier.fit(X_train, y_train)
    print("KNN completed!")
     
    # Train SVM classifier
    svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')  
    svm_classifier.fit(X_train, y_train)
    print("SVM completed!")
    
    # Train Random Forest classifier with different parameters
    rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)  
    rf_classifier.fit(X_train, y_train)
    print("Random Forest training..")
    
    print("Random Forest training completed!")
    return knn_classifier, svm_classifier, rf_classifier

def find_best_features(X_train, X_test, y_train):
    # Train classifiers to find feature importances
    knn_classifier, svm_classifier, rf_classifier = train_classifiers(X_train, y_train)
    
    # Get feature importances from Random Forest classifier
    rf_importances = rf_classifier.feature_importances_
    
    # Sort feature importances in descending order
    sorted_indices = np.argsort(rf_importances)[::-1]
    
    # Select top features
    top_features = [selected_features[i] for i in sorted_indices[:20]]
    
    return top_features


def extract_features(file_path):
    # Load the scaler object
    global scaler
    
    # Load scaler from pickle file
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    if scaler is None:
        raise ValueError("Scaler object not initialized. Make sure to load the dataset before extracting features.")

    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    
    # Extract features
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
    rms = librosa.feature.rms(y=audio)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
    
    # Only extract harmonic component (no need for variances)
    harmony = librosa.effects.harmonic(y=audio)
    harmony_mean = np.mean(harmony)
    
    perceptr = librosa.effects.percussive(y=audio)
    perceptr_mean = np.mean(perceptr)
    
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    
    # Calculate mean and variance of each feature
    features = [
        np.mean(chroma_stft), np.var(chroma_stft),
        np.mean(rms), np.var(rms),
        np.mean(spectral_centroid), np.var(spectral_centroid),
        np.mean(spectral_bandwidth), np.var(spectral_bandwidth),
        np.mean(rolloff), np.var(rolloff),
        np.mean(zero_crossing_rate), np.var(zero_crossing_rate),
        harmony_mean, np.var(harmony),
        perceptr_mean, np.var(perceptr),
        tempo
    ]
    
    for i in range(mfccs.shape[0]):
        features.extend([np.mean(mfccs[i]), np.var(mfccs[i])])
    # Convert features list to NumPy array
    features_array = np.array(features)
    
    # Reshape the NumPy array
    features_reshaped = features_array.reshape(1, -1)
    
    # Normalize the features using the loaded scaler
    features_normalized = scaler.transform(features_reshaped)
    
    return features_normalized.ravel()



def main():
    X_train, X_test, y_train, y_test = load_dataset()
    X_train_processed, X_test_processed = preprocess_data(X_train, X_test, y_train)
    
    # Find best features
    best_features = find_best_features(X_train_processed, X_test_processed, y_train)
    
    print("Best features:", best_features)

if __name__ == "__main__":
    main()
