# augment_data.py
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Augmentation pipeline
augment = A.Compose([
    A.Rotate(limit=10, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=1),
    A.Perspective(p=0.3),
    A.GaussNoise(p=0.4)
])

def augment_images(input_dir, output_dir, augmentations_per_image=20):
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        save_path = os.path.join(output_dir, class_name)
        os.makedirs(save_path, exist_ok=True)

        for img_name in tqdm(os.listdir(class_path), desc=f"Augmenting {class_name}"):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for i in range(augmentations_per_image):
                aug_img = augment(image=image)['image']
                save_name = f"{os.path.splitext(img_name)[0]}_aug{i}.png"
                cv2.imwrite(os.path.join(save_path, save_name), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    input_dir = "data"
    output_dir = "augmented_data"
    augment_images(input_dir, output_dir)
