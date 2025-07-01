#setup.py
from tensorflow.keras.models import load_model

# تحميل الموديل مرة واحدة عند استيراد الملف
MODEL_PATH = 'path/to/brain_classifier_final.h5'  # حط المسار الصحيح هنا
model = load_model(MODEL_PATH)

# لو حابب ترجع اسم الكلاسات (optional)
class_labels = ['Normal', 'Tumor']  # أو حسب ترتيب التدريب
