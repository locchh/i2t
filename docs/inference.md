Here’s how you can load a saved model and use it for inference (generating captions for new images):

---

### 1. **Loading the Model**
You need the same model architecture used during training and the saved model weights.

```python
# Load Model Architecture
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = ImageCaptioningModel(
    encoder=ImageEncoder(embed_size),
    decoder=CaptionDecoder(embed_size, hidden_size, vocab_size)
).to(device)

# Load Saved Weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
```

---

### 2. **Define Inference Function**
This function generates captions for a given image using the loaded model.

```python
def generate_caption(model, image_path, tokenizer, max_length, transform):
    # Preprocess Image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Start Captioning
    with torch.no_grad():
        features = model.encoder(image)
        caption = [tokenizer.cls_token_id]  # Start with <CLS> token
        for _ in range(max_length):
            caption_tensor = torch.tensor(caption).unsqueeze(0).to(device)
            outputs = model.decoder(features, caption_tensor)
            next_token = outputs.argmax(2)[:, -1].item()  # Get the highest-probability token
            caption.append(next_token)
            if next_token == tokenizer.sep_token_id:  # Stop if <SEP> token is generated
                break

    # Convert Token IDs to Text
    generated_caption = tokenizer.decode(caption, skip_special_tokens=True)
    return generated_caption
```

---

### 3. **Run Inference on a New Image**
Here’s how to use the model to generate a caption for a specific image.

```python
# Example Image Path
image_path = "flickr8k/Images/example.jpg"

# Define Tokenizer and Transformations
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Generate Caption
caption = generate_caption(model, image_path, tokenizer, max_length=20, transform=transform)
print(f"Generated Caption: {caption}")
```

---

### 4. **Batch Inference**
If you want to generate captions for multiple images in a folder:
```python
import glob

def batch_inference(model, folder_path, tokenizer, transform, max_length):
    image_paths = glob.glob(f"{folder_path}/*.jpg")
    results = {}
    for img_path in image_paths:
        caption = generate_caption(model, img_path, tokenizer, max_length, transform)
        results[img_path] = caption
    return results

# Run Batch Inference
folder_path = "flickr8k/Images"
results = batch_inference(model, folder_path, tokenizer, transform, max_length=20)

# Print Results
for img, cap in results.items():
    print(f"{img}: {cap}")
```

---

### 5. **Key Notes**
- Ensure the tokenizer and preprocessing steps used during inference match those used during training.
- You can load the model weights into a different device (CPU/GPU) using `map_location`.
- For real-world applications, fine-tuning or improving the inference pipeline (e.g., beam search instead of greedy decoding) can significantly enhance caption quality.

Would you like to try implementing beam search or improving the generation quality?
