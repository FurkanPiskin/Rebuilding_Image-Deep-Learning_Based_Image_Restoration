import os
import random
import shutil
# Dataseti test train val diye klasörlere ayırmak için kullanılmıştır


src_folder = r"" #Kaynak dataset dosyası
dst_folder = r"" #Datasetlerin ayrılacağı hedef klasör

# Bölme oranları
ratios = {
    "face_train": 0.70,
    "face_val":   0.15,
    "face_test":  0.15,
}

# 1) Kaynak klasördeki resimleri topla
all_files = [f for f in os.listdir(src_folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if not all_files:
    raise RuntimeError(f"Kaynak klasörde resim bulunamadı: {src_folder}")

random.shuffle(all_files)
total = len(all_files)

# 2) Sınır indekslerini hesapla
train_end = int(ratios["face_train"] * total)
val_end   = train_end + int(ratios["face_val"] * total)

split_files = {
    "face_train": all_files[:train_end],
    "face_val":   all_files[train_end:val_end],
    "face_test":  all_files[val_end:],
}

# 3) Hedef klasörleri oluştur ve dosyaları kopyala
for split_name, files in split_files.items():
    dest_dir = os.path.join(dst_folder, split_name)
    os.makedirs(dest_dir, exist_ok=True)

    for fname in files:
        src_path = os.path.join(src_folder, fname)
        dst_path = os.path.join(dest_dir, fname)

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)  
            print(f"Kaynak dosya bulunamadı, atlanıyor: {src_path}")

print("✅ Veri başarıyla face_train / face_val / face_test klasörlerine bölündü.")
