import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load the dataset
df = pd.read_csv('bank-full.csv', sep=';')

# 2. Preprocess the target variable (yes=1, no=0)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# 3. Drop 'duration' to prevent data leakage
df_clean = df.drop(columns=['duration'])

# 4. Prepare features x and target y
X = df_clean.drop(columns=['y'])
y = df_clean['y']

# 5. Convert categorical variables into numeric
X_encoded = pd.get_dummies(X, drop_first=True)

# 6. Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# 8. Make predictions
y_pred = clf.predict(X_test)

# 9. Evaluate and print
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# CHART GENERATION FOR DASHBOARD

plt.style.use('ggplot')

# Chart 1: Confusion matrix heatmap
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Purchase', 'Purchase'],
            yticklabels=['No Purchase', 'Purchase'])
plt.title('Model Confusion Matrix', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('decision_tree_confusion_matrix.png', dpi=150)
plt.close()

# Chart 2: Top 10 feature importances
plt.figure(figsize=(10, 6))
importances = pd.Series(clf.feature_importances_, index=X_encoded.columns)
top_features = importances.sort_values(ascending=True).tail(10)
top_features.plot(kind='barh', color='#8b5cf6')
plt.title('Top 10 Most Predictive Behavioral Drivers', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Relative Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('decision_tree_features.png', dpi=150)
plt.close()

# Chart 3: Tree structure layout
plt.figure(figsize=(24, 12))
plot_tree(clf,
          feature_names=X_encoded.columns,
          class_names=['No Purchase', 'Purchase'],
          filled=True,
          rounded=True,
          fontsize=10,
          max_depth=3) # Showing up to depth 3 for structural legibility
plt.title('Decision Tree Classification Path Matrix (Truncated View)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('decision_tree_structure.png', dpi=200)
plt.close()
