import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

# Chemin du dossier contenant les images originales
input_dir = 'Images'
# Chemin du dossier où les images augmentées seront sauvegardées
output_dir = 'Images'
os.makedirs(output_dir, exist_ok=True)

# Création de l'objet de génération de données avec différentes transformations
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,  
    brightness_range=(0.9, 1.1),  # Ajustement de la luminosité
    channel_shift_range=10,  # Décalage des canaux
    fill_mode='nearest'
)

try:
    # Calcul du nombre d'images à générer par image originale
    total_images_needed = 4000
    current_image_count = len(os.listdir(input_dir))
    images_per_file = (total_images_needed - current_image_count) // current_image_count + 1

    # Processus d'augmentation de données
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Erreur lors de la lecture du fichier {filename}. Ce fichier sera ignoré.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.reshape((1,) + image.shape)  # Redimensionnement nécessaire pour la génération

            # Création des images augmentées
            save_prefix = f"aug_{filename.split('.')[0]}"
            i = 0
            for batch in datagen.flow(image, batch_size=1, save_to_dir=output_dir, save_prefix=save_prefix, save_format='jpeg'):
                i += 1
                if i >= images_per_file:
                    break

    print("Augmentation des données terminée.")
except Exception as e:
    print(f"Une erreur est survenue : {e}")