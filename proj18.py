import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a dummy dataset.csv file if it doesn't exist
try:
    df = pd.read_csv('notebooks/dataset.csv')
except FileNotFoundError:
    print("File 'dataset.csv' not found. Creating a dummy dataset.")
    data = {
        'feature1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    df.to_csv('dataset.csv', index=False)
    print("Dummy 'dataset.csv' created. Please upload your actual file or ensure it's in the correct directory.")

# 1. Общая информация
print(df.shape)          # размер
print(df.dtypes)         # типы
print(df.isnull().sum()) # пропуски
print(df.describe())     # статистика

# 2. Распределение целевой переменной
plt.figure(figsize=(6,4))
df['target'].value_counts().plot(kind='bar')
plt.title('Распределение классов')
plt.savefig('class_dist.png', dpi=150)

# 3. Корреляционная матрица
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.savefig('correlation.png', dpi=150)

# Анализ размеров изображений
sizes = []
for f in os.listdir('images/'):
    img = Image.open(f'images/{f}')
    sizes.append(img.size)

widths, heights = zip(*sizes)
print(f'Мин: {min(widths)}x{min(heights)}, Макс: {max(widths)}x{max(heights)}')

# Показ примеров по классам
fig, axes = plt.subplots(2, 5, figsize=(15,6))
for ax, class_name in zip(axes.flat, class_names[:10]):
    sample = random.choice(os.listdir(f'data/{class_name}'))
    ax.imshow(Image.open(f'data/{class_name}/{sample}'))
    ax.set_title(class_name); ax.axis('off')
plt.savefig('samples_grid.png', dpi=150)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(42)  # ОБЯЗАТЕЛЬНО!

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

baseline = LogisticRegression(max_iter=1000, random_state=42)
baseline.fit(X_train, y_train)

y_pred = baseline.predict(X_test)
print(classification_report(y_test, y_pred))

# Сохраните результаты!
baseline_f1 = f1_score(y_test, y_pred, average='macro')
print(f'Baseline F1-macro: {baseline_f1:.4f}')

