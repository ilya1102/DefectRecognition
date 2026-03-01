
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import json
import os

# Параметры
IMG_SIZE = (224, 224)      # размер, который ожидает MobileNetV2
BATCH_SIZE = 32             # количество изображений за один шаг
EPOCHS = 10                 # количество эпох обучения
NUM_CLASSES = 2             # у нас два класса
TRAIN_DIR = 'data/train'    # путь к обучающим данным
VAL_DIR = 'data/val'        # путь к валидационным данным

# 1. Подготовка данных с аугментацией
train_datagen = ImageDataGenerator(
    rescale=1./255,                # нормализация пикселей к [0,1]
    rotation_range=20,              # случайный поворот до 20 градусов
    width_shift_range=0.2,          # горизонтальный сдвиг
    height_shift_range=0.2,         # вертикальный сдвиг
    horizontal_flip=True,           # отражение по горизонтали
    zoom_range=0.2                   # случайное приближение
)

# Для валидации только масштабируем (без аугментации)
val_datagen = ImageDataGenerator(rescale=1./255)

# Загружаем изображения из папок
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# 2. Создание модели на основе предобученной MobileNetV2
# Загружаем базовую модель без верхушки (include_top=False)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Замораживаем все слои базовой модели (они уже обучены)
base_model.trainable = False

# Добавляем новые слои классификации
x = base_model.output
x = GlobalAveragePooling2D()(x)          # усреднение пространственных признаков
x = Dense(128, activation='relu')(x)     # полносвязный слой для комбинации признаков
predictions = Dense(NUM_CLASSES, activation='softmax')(x)  # выходной слой с 2 классами

# Составляем полную модель
model = Model(inputs=base_model.input, outputs=predictions)

# 3. Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Обучение
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

# 5. Сохранение модели
model.save('defect_model.h5')
print("✅ Модель сохранена как defect_model.h5")

# 6. Сохранение словаря классов
# flow_from_directory создаёт соответствие: имя папки -> индекс
class_indices = train_generator.class_indices
# Переворачиваем: индекс -> имя класса
class_names = {str(v): k for k, v in class_indices.items()}
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print("✅ Имена классов сохранены в class_names.json")
print("Соответствие индексов и классов:", class_indices)