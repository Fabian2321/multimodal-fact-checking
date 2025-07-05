#!/usr/bin/env python3
"""
Create Few-Shot Example Images for BLIP2 Experiments
Generates simple example images for few-shot learning.
"""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_dog_on_couch_image():
    """Create a simple image of a dog on a couch."""
    # Create a 224x224 image (standard size for many models)
    img = Image.new('RGB', (224, 224), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple couch (brown rectangle)
    draw.rectangle([20, 140, 204, 200], fill='brown', outline='saddlebrown', width=2)
    
    # Draw couch cushions
    draw.rectangle([25, 145, 95, 195], fill='beige', outline='brown', width=1)
    draw.rectangle([105, 145, 175, 195], fill='beige', outline='brown', width=1)
    
    # Draw a simple dog (brown oval with ears)
    # Dog body
    draw.ellipse([80, 100, 140, 140], fill='brown', outline='saddlebrown', width=2)
    # Dog head
    draw.ellipse([90, 80, 130, 110], fill='brown', outline='saddlebrown', width=2)
    # Dog ears
    draw.ellipse([85, 75, 105, 95], fill='saddlebrown')
    draw.ellipse([115, 75, 135, 95], fill='saddlebrown')
    # Dog eyes
    draw.ellipse([95, 85, 105, 95], fill='black')
    draw.ellipse([115, 85, 125, 95], fill='black')
    # Dog nose
    draw.ellipse([108, 95, 112, 99], fill='black')
    
    return img

def create_cat_playing_piano_image():
    """Create a simple image of a cat playing piano."""
    # Create a 224x224 image
    img = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a piano (black rectangle with white keys)
    draw.rectangle([30, 120, 194, 200], fill='black', outline='black', width=2)
    
    # Draw piano keys (white rectangles)
    for i in range(7):
        x = 35 + i * 22
        draw.rectangle([x, 125, x + 20, 195], fill='white', outline='black', width=1)
    
    # Draw a simple cat (gray oval with ears)
    # Cat body
    draw.ellipse([100, 60, 140, 100], fill='gray', outline='dimgray', width=2)
    # Cat head
    draw.ellipse([105, 40, 135, 70], fill='gray', outline='dimgray', width=2)
    # Cat ears
    draw.polygon([(105, 40), (100, 30), (110, 40)], fill='gray')
    draw.polygon([(135, 40), (140, 30), (130, 40)], fill='gray')
    # Cat eyes
    draw.ellipse([108, 45, 115, 52], fill='green')
    draw.ellipse([125, 45, 132, 52], fill='green')
    # Cat nose
    draw.ellipse([118, 50, 122, 54], fill='pink')
    
    return img

def main():
    """Create and save the few-shot example images."""
    # Create output directory
    output_dir = "data/few_shot_examples"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating few-shot example images...")
    
    # Create dog on couch image (real example)
    dog_img = create_dog_on_couch_image()
    dog_path = os.path.join(output_dir, "real_example.jpg")
    dog_img.save(dog_path, "JPEG", quality=95)
    print(f"Created real example: {dog_path}")
    
    # Create cat playing piano image (fake example)
    cat_img = create_cat_playing_piano_image()
    cat_path = os.path.join(output_dir, "fake_example.jpg")
    cat_img.save(cat_path, "JPEG", quality=95)
    print(f"Created fake example: {cat_path}")
    
    print("\nFew-shot example images created successfully!")
    print("These images will be used for BLIP2 few-shot experiments.")
    print("\nImage descriptions:")
    print("- real_example.jpg: A dog sitting on a couch (matches text)")
    print("- fake_example.jpg: A cat playing piano (doesn't match text)")

if __name__ == "__main__":
    main() 