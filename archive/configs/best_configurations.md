# Best Performing Model Configurations

## CLIP ViT-Base/16

### Zero-Shot Configuration (79.0% Accuracy)
- Architecture: ViT-Base/16
- Input Resolution: 224x224
- Embedding Dimension: 512
- Similarity Metric: Cosine Similarity
- Threshold: 0.272 (optimized)

**Implementation Details:**
```python
def zero_shot_clip(image, text):
    # Encode image and text
    image_features = clip_model.encode_image(image)
    text_features = clip_model.encode_text(text)
    
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity
    similarity = (image_features @ text_features.T).item()
    
    return similarity > 0.272
```

### Optimized Configuration (82.0% Accuracy)

### Model Configuration
- Architecture: ViT-Base/16
- Input Resolution: 224x224
- Embedding Dimension: 512
- Similarity Metric: Cosine Similarity

### Preprocessing Pipeline
- Multi-crop Strategy: 3 crops (center + corners)
- Text Preprocessing: Stopword removal, lowercasing
- Image Normalization: ImageNet stats
- L2 Normalization: Applied to embeddings

### Optimal Parameters
- Similarity Threshold: 0.272
- Batch Size: 32
- Device: NVIDIA A100-SXM4-40GB
- Processing Speed: 0.1s per sample

### RAG Configuration
- Knowledge Base: Fact-checking guidelines
- Retrieval: Top-3 documents
- Similarity Threshold: 0.7
- Impact: +3.0% accuracy improvement

### Performance Comparison
| Metric | Zero-Shot | Optimized | Improvement |
|--------|-----------|-----------|-------------|
| Accuracy | 79.0% | 82.0% | +3.0% |
| Precision | 80.9% | 80.9% | +0.0% |
| Recall | 76.0% | 76.0% | +0.0% |
| F1-Score | 78.4% | 78.4% | +0.0% |
| ROC AUC | 81.4% | 81.4% | +0.0% |

## BLIP2 OPT-2.7B

### Zero-Shot Configuration (56.0% Accuracy)
- Base Model: Salesforce/blip2-opt-2.7b
- Input Resolution: 224x224
- Max Text Length: 75 tokens
- Output Format: Binary classification

**Prompt Template:**
```python
zero_shot_template = """
Question: Does the text match the image? Answer with only 'yes' or 'no'.

Text: {text}
"""
```

**Implementation Details:**
```python
def zero_shot_blip2(image, text):
    inputs = processor(
        images=image,
        text=zero_shot_template.format(text=text),
        return_tensors="pt"
    ).to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=75,
        temperature=1.0,
        top_k=10,
        do_sample=False
    )
    
    return processor.decode(outputs[0], skip_special_tokens=True)
```

### Few-Shot Configuration (62.0% Accuracy)

### Model Configuration
- Base Model: LLaVA-1.5-7B
- Input Resolution: 224x224
- Max Text Length: 512 tokens
- Output Format: Binary classification

### Preprocessing Pipeline
- Prompt Template: llava_match_metadata
- Metadata Integration: Yes
- Enhanced Parsing: Yes
- Batch Size: 1

### Optimal Parameters
- Temperature: 0.7
- Max New Tokens: 20
- Device: NVIDIA A100-SXM4-40GB
- Processing Speed: 0.930s per sample

### RAG Configuration
- Knowledge Base: Fact-checking guidelines
- Prompt Enhancement: Yes
- Impact: +2.0% accuracy improvement

**Few-Shot Examples Implementation:**
```python
def get_few_shot_examples():
    return [
        {
            "image_desc": "Cat on red chair",
            "text": "A cat sitting on a red chair",
            "label": "yes",
            "explanation": "The image shows exactly what the text describes"
        },
        {
            "image_desc": "Ocean sunset",
            "text": "Beautiful sunset over the ocean",
            "label": "yes",
            "explanation": "The image matches the described scene"
        },
        {
            "image_desc": "Generic space photo",
            "text": "NASA discovers alien life on Mars",
            "label": "no",
            "explanation": "The text makes claims not supported by the image"
        },
        {
            "image_desc": "Political rally",
            "text": "President Obama riding a unicorn",
            "label": "no",
            "explanation": "The text describes an impossible scenario"
        }
    ]

def format_few_shot_prompt(examples, query_image, query_text):
    prompt = "Given an image and text, determine if they match. Here are some examples:\n\n"
    
    for i, ex in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Image shows: {ex['image_desc']}\n"
        prompt += f"Text: {ex['text']}\n"
        prompt += f"Match: {ex['label']}\n"
        prompt += f"Explanation: {ex['explanation']}\n\n"
    
    prompt += "Now, for this new case:\n"
    prompt += f"Text: {query_text}\n"
    prompt += "Does the text match the image? Answer with only 'yes' or 'no'."
    
    return prompt

### Performance Comparison
| Metric | Zero-Shot | Few-Shot | Improvement |
|--------|-----------|----------|-------------|
| Accuracy | 56.0% | 62.0% | +6.0% |
| Precision | 54.3% | 57.1% | +2.8% |
| Recall | 54.2% | 20.0% | -34.2% |
| F1-Score | 54.0% | 29.6% | -24.4% |
| ROC AUC | 42.8% | 42.8% | +0.0% |

## LLaVA 1.5-7B

### Zero-Shot Configuration (63.0% Accuracy)
- Base Model: LLaVA-1.5-7B
- Vision Encoder: CLIP ViT-L/14
- Language Model: Vicuna-7B-v1.5
- Max Text Length: 512 tokens

**Prompt Templates:**
```python
# Simple zero-shot
zero_shot_simple = "Does this image match the text: '{text}'? Answer with only 'yes' or 'no'."

# Detailed zero-shot
zero_shot_detailed = """
Analyze this image and text pair:
Text: {text}

Step by step:
1. What do you see in the image?
2. What does the text claim?
3. Do they match?

Answer with only 'yes' or 'no'.
"""

# Metadata-aware zero-shot
zero_shot_metadata = """
Context:
- Text: {text}
- Source: {metadata['subreddit']}
- Domain: {metadata['domain']}

Based on the image and context, is this a genuine match? Answer with only 'yes' or 'no'.
"""
```

### Few-Shot Configuration (65.0% Accuracy)

### Model Configuration
- Base Model: LLaVA-1.5-7B
- Input Resolution: 224x224
- Max Text Length: 512 tokens
- Output Format: Binary classification

### Preprocessing Pipeline
- Prompt Template: llava_match_metadata
- Metadata Integration: Yes
- Enhanced Parsing: Yes
- Batch Size: 1

### Optimal Parameters
- Temperature: 0.7
- Max New Tokens: 20
- Device: NVIDIA A100-SXM4-40GB
- Processing Speed: 0.930s per sample

### RAG Configuration
- Knowledge Base: Fact-checking guidelines
- Prompt Enhancement: Yes
- Impact: +2.0% accuracy improvement

**Chain-of-Thought Implementation:**
```python
def format_cot_prompt(examples, query_image, query_text):
    prompt = "Let's analyze image-text pairs step by step. Here are some examples:\n\n"
    
    for i, ex in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Image content: {ex['image_desc']}\n"
        prompt += f"Text claim: {ex['text']}\n"
        prompt += "Analysis:\n"
        prompt += "1. Image shows: " + ex['analysis']['image_content'] + "\n"
        prompt += "2. Text claims: " + ex['analysis']['text_claim'] + "\n"
        prompt += "3. Comparison: " + ex['analysis']['comparison'] + "\n"
        prompt += f"Conclusion: {ex['label']}\n\n"
    
    prompt += "Now analyze this case:\n"
    prompt += f"Text: {query_text}\n"
    prompt += "Follow the same steps and conclude with only 'yes' or 'no'."
    
    return prompt
```

### Performance Comparison
| Metric | Zero-Shot | Few-Shot | Improvement |
|--------|-----------|----------|-------------|
| Accuracy | 63.0% | 65.0% | +2.0% |
| Precision | 64.7% | 63.6% | -1.1% |
| Recall | 66.0% | 70.0% | +4.0% |
| F1-Score | 65.3% | 66.7% | +1.4% |
| Processing Time | 0.93s | 1.10s | +0.17s |

## Processing Pipeline Details

### Image Processing Pipeline
```python
class ImageProcessor:
    def __init__(self, model_type):
        self.model_type = model_type
        self.transforms = self._get_transforms()
    
    def _get_transforms(self):
        if self.model_type == "clip":
            return T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), 
                          (0.229, 0.224, 0.225))
            ])
        elif self.model_type == "blip2":
            return T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073),
                          (0.26862954, 0.26130258, 0.27577711))
            ])
        elif self.model_type == "llava":
            return T.Compose([
                T.Resize((336, 336)),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073),
                          (0.26862954, 0.26130258, 0.27577711))
            ])

    def process_image(self, image_path, multi_crop=False):
        image = Image.open(image_path).convert('RGB')
        if not multi_crop:
            return self.transforms(image)
        
        crops = []
        # Center crop
        crops.append(self.transforms(image))
        
        # Corner crops
        size = image.size
        corner_size = int(min(size) * 0.8)
        corners = [
            (0, 0),
            (size[0]-corner_size, 0),
            (0, size[1]-corner_size),
            (size[0]-corner_size, size[1]-corner_size)
        ]
        
        for corner in corners:
            crop = image.crop((
                corner[0],
                corner[1],
                corner[0] + corner_size,
                corner[1] + corner_size
            ))
            crops.append(self.transforms(crop))
        
        return torch.stack(crops)
```

### Text Processing Pipeline
```python
class TextProcessor:
    def __init__(self, model_type):
        self.model_type = model_type
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english'))
    
    def process_text(self, text, clean_level='basic'):
        if clean_level == 'basic':
            return self._basic_clean(text)
        elif clean_level == 'advanced':
            return self._advanced_clean(text)
        else:
            return text
    
    def _basic_clean(self, text):
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        return text.lower().strip()
    
    def _advanced_clean(self, text):
        # Basic cleaning
        text = self._basic_clean(text)
        # Tokenize and remove stopwords
        doc = self.nlp(text)
        # Lemmatize and filter
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
        ]
        return ' '.join(tokens)
    
    def get_variants(self, text):
        variants = []
        # Original text
        variants.append(text)
        # Basic cleaned
        variants.append(self._basic_clean(text))
        # Advanced cleaned
        variants.append(self._advanced_clean(text))
        # Keyword extraction
        doc = self.nlp(text)
        keywords = ' '.join([
            token.text for token in doc
            if token.pos_ in ['NOUN', 'VERB', 'ADJ']
        ])
        variants.append(keywords)
        return variants
``` 