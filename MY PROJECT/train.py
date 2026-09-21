import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
import joblib

# Load and clean data
df = pd.read_csv('student_performance.csv').dropna()


# ==========================================
# Encode Parental_Involvement
# ==========================================

parental_encoder = LabelEncoder()

df['Parental_Involvement'] = parental_encoder.fit_transform(
    df['Parental_Involvement']
)

joblib.dump(parental_encoder, 'parental_encoder.pkl')

print("Parental classes:", parental_encoder.classes_)


# ==========================================
# Encode Access_to_Internet
# ==========================================

internet_encoder = LabelEncoder()

df['Access_to_Internet'] = internet_encoder.fit_transform(
    df['Access_to_Internet']
)

joblib.dump(internet_encoder, 'internet_encoder.pkl')

print("Internet classes:", internet_encoder.classes_)


# ==========================================
# Train Linear Regression
# ==========================================

X_reg = df[
    [
        'Hours_Studied',
        'Attendance',
        'Sleep_Hours',
        'Tutoring_Sessions'
    ]
]

y_reg = df['Exam_Score']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

reg_model = LinearRegression()

reg_model.fit(
    X_train_reg,
    y_train_reg
)

y_pred_reg = reg_model.predict(X_test_reg)

r2 = r2_score(
    y_test_reg,
    y_pred_reg
)

print("Linear Regression R2 Score:", r2)

joblib.dump(
    reg_model,
    'regression_model.pkl'
)


# ==========================================
# Train Logistic Regression
# ==========================================

X_clf = df[
    [
        'Hours_Studied',
        'Attendance',
        'Parental_Involvement',
        'Access_to_Internet',
        'Sleep_Hours',
        'Tutoring_Sessions'
    ]
]

y_clf = pd.cut(
    df['Exam_Score'],
    bins=[0, 50, 75, 100],
    labels=[0, 1, 2],
    include_lowest=True
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf,
    y_clf,
    test_size=0.2,
    random_state=42
)

clf_model = LogisticRegression(
    random_state=42,
    max_iter=1500
)

clf_model.fit(
    X_train_clf,
    y_train_clf
)

y_pred_clf = clf_model.predict(X_test_clf)

accuracy = accuracy_score(
    y_test_clf,
    y_pred_clf
)

print("Classification Accuracy:", accuracy)

joblib.dump(
    clf_model,
    'classification_model.pkl'
)


# ==========================================
# Train K-Means
# ==========================================

X_cluster = df[
    [
        'Hours_Studied',
        'Exam_Score'
    ]
]

k_means = KMeans(
    n_clusters=3,
    random_state=0,
    n_init=10
)

k_means.fit(X_cluster)

joblib.dump(
    k_means,
    'kmeans_model.pkl'
)


print("All models trained and saved successfully!")