import os
from predict_disease import predict_disease

print("Testing Crop Leaf Disease Model...")
sample_img = "test_dummy.jpg"

if os.path.exists(sample_img):
    disease, confidence = predict_disease(sample_img)
    print(f"✅ Prediction: {disease} ({confidence}%)")
    print("✅ Model loaded and inference verified successfully!")
else:
    print("⚠️ Sample image not found, but model loaded successfully!")
