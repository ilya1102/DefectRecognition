import os
import json
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Определяем путь к корневой папке проекта (где лежат defect_model.h5 и class_names.json)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'defect_model.h5')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'class_names.json')

# Настройки приложения
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'          # папка для временного хранения
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # макс. размер файла 16 МБ

# Загружаем модель и словарь классов
model = load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)   # например {"0": "silk_spot", "1": "waist_folding"}

def predict_image(img_path):
    """Загружает изображение, предсказывает класс и возвращает топ-3 вероятности"""
    # Загружаем и подготавливаем изображение
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0   # нормализация

    # Получаем предсказания
    predictions = model.predict(img_array)[0]   # массив из двух вероятностей

    # Сортируем по убыванию вероятности и берём топ-3 (все два)
    top3_idx = np.argsort(predictions)[-3:][::-1]   # индексы от наибольшего к наименьшему
    top3 = []
    for idx in top3_idx:
        name = class_names[str(idx)]
        prob = float(predictions[idx])               # преобразуем в обычный float
        top3.append((name, prob))
    return top3

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Проверяем, прикреплён ли файл
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Сохраняем файл во временную папку
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Получаем предсказания
            top3 = predict_image(filepath)

            # Удаляем временный файл (чтобы не засорять)
            os.remove(filepath)

            # Передаём результаты в шаблон
            return render_template('result.html', predictions=top3)
    return render_template('index.html')

if __name__ == '__main__':
    # Убедимся, что папка для загрузок существует
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)