from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from playsound import playsound
import os
import time

# Charger le modèle et le processeur
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# Configurer la caméra (utilisez 0 pour la caméra par défaut)
cap = cv2.VideoCapture(0)

# Définir l'intervalle entre les captures d'images en secondes
capture_interval = 3

while True:
    # Capture une image de la caméra
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from the camera.")
        break

    # Convertir l'image OpenCV en format PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Effectuer la détection d'objets avec DETR
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Convertir les résultats en format COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

    # Afficher l'image avec les boîtes englobantes
    fig, ax = plt.subplots(1)
    ax.imshow(frame)

    print("•")

    # Afficher les informations sur les objets détectés
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        class_name = model.config.id2label[label.item()]

        # Create a rectangle patch
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r',
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Display class label and confidence score
        ax.text(box[0], box[1], f'{class_name} {round(score.item(), 3)}', color='r', verticalalignment='top')

        # Check if the detected object is a phone
        if class_name.lower() == "cell phone":
            # Format the image name
            image_name = f"{class_name.lower()}_{round(score.item() * 100, 2)}%_{box}.jpg"
            playsound("alarm.mp3")
            alarm_triggered = True
            # Save the image with the formatted name
            image.save(os.path.join('History of offencces/', image_name))
            print("PHONE : Picture downloaded into 'History of offencces'!")
            break

        # Check if the detected object is a bottle
        if class_name.lower() == "bottle":
            # Format the image name
            image_name = f"{class_name.lower()}_{round(score.item() * 100, 2)}%_{box}.jpg"
            playsound("alarm.mp3")
            alarm_triggered = True
            # Save the image with the formatted name
            image.save(os.path.join('History of offencces/', image_name))
            print("BOTTLE : Picture downloaded into 'History of offencces'!")
            break

        # Check if the detected object is a Cigarette
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Définir les plages de couleurs
        lower_rgb_cigarette = np.array([200, 100, 20])
        upper_rgb_cigarette = np.array([255, 180, 70])

        mask = cv2.inRange(image_rgb, lower_rgb_cigarette, upper_rgb_cigarette)

        if np.any(mask) or class_name.lower() == "toothbrush":
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Dessiner les contours sur l'image d'origine
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            image_name = f"Cigarette_{round(score.item() * 100, 2)}%_{box}.jpg"
            playsound("alarm.mp3")
            alarm_triggered = True
            # Save the image with the formatted name
            image.save(os.path.join('History of offencces/', image_name))
            print("CIGARETTE : Picture downloaded into 'History of offencces'!")
            cv2.imshow('Detected Object', frame)
            #plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #plt.show()
            break

    # Afficher l'image avec les résultats
    #plt.show()

    # Attendre l'intervalle spécifié avant de capturer la prochaine image
    #time.sleep(capture_interval)

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
