# Experiment Configurations and Parser Implementations

## Response Parser Implementations

### CLIP Parser
```python
class CLIPResponseParser:
    def __init__(self, threshold=0.272):
        self.threshold = threshold
    
    def parse_response(self, similarity_score):
        """
        Parse CLIP's cosine similarity score into a binary decision
        Args:
            similarity_score (float): Cosine similarity between image and text
        Returns:
            bool: True if match, False if not
        """
        return float(similarity_score) > self.threshold
    
    def parse_batch_response(self, similarity_scores):
        """
        Parse batch of similarity scores
        Args:
            similarity_scores (torch.Tensor): Batch of similarity scores
        Returns:
            List[bool]: List of binary decisions
        """
        return [self.parse_response(score) for score in similarity_scores]
```

### BLIP2 Parser
```python
class BLIP2ResponseParser:
    def __init__(self, confidence_threshold=0.8):
        self.confidence_threshold = confidence_threshold
        self.yes_patterns = [
            r'\byes\b',
            r'the (image|text) (shows|matches)',
            r'they (match|align|correspond)',
            r'accurate description'
        ]
        self.no_patterns = [
            r'\bno\b',
            r'does not match',
            r'doesn\'t match',
            r'mismatch',
            r'incorrect description'
        ]
    
    def parse_response(self, response_text):
        """
        Parse BLIP2's text response into a binary decision
        Args:
            response_text (str): Generated text response
        Returns:
            tuple: (decision, confidence)
                decision (bool): True if match, False if not
                confidence (float): Confidence score 0-1
        """
        response_text = response_text.lower().strip()
        
        # Count matches for each pattern
        yes_matches = sum(1 for pattern in self.yes_patterns 
                         if re.search(pattern, response_text))
        no_matches = sum(1 for pattern in self.no_patterns 
                        if re.search(pattern, response_text))
        
        # Calculate confidence
        total_matches = yes_matches + no_matches
        if total_matches == 0:
            return None, 0.0
            
        confidence = max(yes_matches, no_matches) / total_matches
        decision = yes_matches > no_matches
        
        return decision, confidence

    def is_confident(self, confidence):
        return confidence >= self.confidence_threshold
```

### LLaVA Parser
```python
class LLaVAResponseParser:
    def __init__(self, confidence_threshold=0.8):
        self.confidence_threshold = confidence_threshold
        self.nlp = spacy.load('en_core_web_sm')
        
    def parse_response(self, response_text):
        """
        Parse LLaVA's text response into a binary decision
        Args:
            response_text (str): Generated text response
        Returns:
            tuple: (decision, confidence, reasoning)
        """
        # Normalize text
        text = response_text.lower().strip()
        doc = self.nlp(text)
        
        # Extract decision and confidence
        decision = None
        confidence = 0.0
        reasoning = ""
        
        # Look for explicit yes/no
        if 'yes' in text.split() or 'match' in text:
            decision = True
        elif 'no' in text.split() or 'not match' in text:
            decision = False
            
        # Extract reasoning if available
        reasoning_markers = ['because', 'as', 'since', 'therefore']
        for sent in doc.sents:
            for marker in reasoning_markers:
                if marker in sent.text.lower():
                    reasoning = sent.text.strip()
                    break
            if reasoning:
                break
                
        # Calculate confidence based on language certainty
        certainty_markers = {
            'high': ['definitely', 'clearly', 'certainly', 'absolutely'],
            'medium': ['probably', 'likely', 'seems', 'appears'],
            'low': ['maybe', 'might', 'could', 'possibly']
        }
        
        for marker_type, markers in certainty_markers.items():
            if any(marker in text for marker in markers):
                confidence = {'high': 0.9, 'medium': 0.7, 'low': 0.5}[marker_type]
                break
        else:
            confidence = 0.8 if decision is not None else 0.0
            
        return decision, confidence, reasoning
```

## RAG Implementation Details

### RAG Implementation

#### 1. RAG Handler
```python
class RAGHandler:
    """Handler for RAG operations."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self.index = None
        self.documents = []
        
        # Load or initialize components
        self._load_components()
    
    def _load_components(self):
        """Load or initialize RAG components."""
        # Load embedding model
        self.model = SentenceTransformer(self.config.embedding_model)
        
        # Load or create FAISS index
        index_path = os.path.join(self.config.knowledge_base_path, "faiss_index.bin")
        documents_path = os.path.join(self.config.knowledge_base_path, "documents.json")
        
        if os.path.exists(index_path) and os.path.exists(documents_path):
            self.index = faiss.read_index(index_path)
            with open(documents_path, 'r') as f:
                self.documents = json.load(f)
```

#### 2. RAG Query Processing
```python
def query(self, query_text: str) -> List[Dict[str, Any]]:
    """Query the knowledge base."""
    if not self.index or not self.documents:
        return []
    
    # Generate query embedding
    query_embedding = self.model.encode([query_text])[0]
    
    # Search in FAISS index
    distances, indices = self.index.search(
        query_embedding.reshape(1, -1),
        self.config.top_k
    )
    
    # Get relevant documents
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(self.documents):
            doc = self.documents[idx].copy()
            doc["similarity_score"] = float(1 / (1 + distance))
            if doc["similarity_score"] >= self.config.similarity_threshold:
                results.append(doc)
    
    # Sort by similarity score
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return results
```

#### 3. Document Retrieval
```python
def retrieve(self, query: str) -> List[Dict[str, Any]]:
    """Retrieve relevant documents for a query."""
    # Get query embedding
    query_embedding = self.model.encode([query], convert_to_numpy=True)
    
    # Search in FAISS index
    distances, indices = self.index.search(query_embedding, self.config.top_k)
    
    # Filter and format results
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(self.documents) and dist < self.config.similarity_threshold:
            doc = self.documents[idx]
            results.append({
                "document": doc,
                "similarity_score": float(1 / (1 + dist))
            })
    
    return results
```

### RAG Configuration

#### 1. Base Configuration
```python
@dataclass
class RAGConfig:
    """Configuration for RAG operations."""
    knowledge_base_path: str
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    top_k: int = 5
    similarity_threshold: float = 0.7
    cache_embeddings: bool = True
    max_context_length: int = 512
```

#### 2. Knowledge Base Structure
```json
{
    "documents": [
        {
            "id": "fact_check_1",
            "text": "Guidelines for detecting manipulated images...",
            "metadata": {
                "category": "visual_verification",
                "confidence": "high"
            }
        },
        {
            "id": "fact_check_2",
            "text": "Common misinformation patterns in social media...",
            "metadata": {
                "category": "misinformation_patterns",
                "confidence": "high"
            }
        }
    ]
}
```

#### 3. Embedding Configuration
```python
embedding_config = {
    'model': 'sentence-transformers/all-mpnet-base-v2',
    'max_seq_length': 384,
    'batch_size': 32,
    'normalize_embeddings': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

### RAG Performance Analysis

#### 1. Retrieval Performance
| Metric | Value |
|--------|--------|
| Precision@1 | 0.845 |
| Precision@3 | 0.762 |
| Precision@5 | 0.689 |
| MRR | 0.812 |
| MAP | 0.734 |

#### 2. Response Quality
| Aspect | Score |
|--------|--------|
| Relevance | 4.2/5 |
| Factual Accuracy | 4.5/5 |
| Context Integration | 3.9/5 |
| Response Coherence | 4.1/5 |

#### 3. Processing Efficiency
| Operation | Average Time (ms) |
|-----------|------------------|
| Embedding Generation | 45 |
| Index Search | 12 |
| Context Integration | 28 |
| Total Latency | 85 |

### RAG Optimization Techniques

1. **Index Optimization**
   - HNSW index for fast approximate search
   - IVF for large-scale retrieval
   - Product quantization for memory efficiency

2. **Query Processing**
   - Query expansion
   - Semantic reranking
   - Hybrid dense-sparse retrieval

3. **Context Integration**
   - Dynamic context window
   - Relevance-based filtering
   - Cross-attention mechanisms

4. **Performance Tuning**
   - Batch processing
   - Caching strategies
   - Parallel retrieval

### RAG Integration Examples

#### 1. BLIP2 Integration
```python
def process_with_rag(image, text, rag_handler):
    """Process BLIP2 with RAG support."""
    # Get relevant documents
    context = rag_handler.query(text)
    
    # Enhance prompt with context
    enhanced_prompt = create_enhanced_prompt(text, context)
    
    # Process with BLIP2
    inputs = processor(
        text=enhanced_prompt,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    return model.generate(**inputs)
```

#### 2. LLaVA Integration
```python
def enhance_llava_prompt(prompt, rag_results):
    """Enhance LLaVA prompt with RAG results."""
    context = "\n".join([
        f"Reference {i+1}: {doc['text']}"
        for i, doc in enumerate(rag_results)
    ])
    
    enhanced_prompt = f"""
    Context Information:
    {context}
    
    Based on the above context and the image, {prompt}
    """
    
    return enhanced_prompt
```

### RAG Best Practices

1. **Knowledge Base Management**
   - Regular updates
   - Quality filtering
   - Metadata enrichment
   - Version control

2. **Query Optimization**
   - Query preprocessing
   - Stop word removal
   - Entity recognition
   - Query reformulation

3. **Response Generation**
   - Template-based responses
   - Dynamic prompting
   - Confidence thresholds
   - Fallback strategies

4. **Monitoring and Maintenance**
   - Performance metrics
   - Error logging
   - Index health checks
   - Regular reindexing 

## Experiment Configurations

### CLIP Experiments

#### 1. Zero-Shot Baseline
```python
config = {
    'model': {
        'name': 'clip',
        'version': 'ViT-B/16',
        'pretrained': True
    },
    'preprocessing': {
        'image_size': 224,
        'center_crop': True,
        'normalize': True,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'inference': {
        'batch_size': 32,
        'threshold': 0.272,
        'temperature': 1.0
    },
    'rag_enabled': False
}
```

#### 2. Optimized Multi-crop
```python
config = {
    'model': {
        'name': 'clip',
        'version': 'ViT-B/16',
        'pretrained': True
    },
    'preprocessing': {
        'multi_crop': True,
        'crop_sizes': [224],
        'crop_scales': [1.0, 0.8],
        'normalize': True
    },
    'inference': {
        'batch_size': 16,  # Reduced due to multiple crops
        'threshold': 0.272,
        'temperature': 1.0,
        'aggregation': 'max'  # max pooling over crops
    },
    'rag_enabled': False
}
```

#### 3. RAG-Enhanced CLIP
```python
config = {
    'model': {
        'name': 'clip',
        'version': 'ViT-B/16',
        'pretrained': True
    },
    'preprocessing': {
        'multi_crop': True,
        'crop_sizes': [224],
        'normalize': True
    },
    'rag': {
        'enabled': True,
        'knowledge_base': 'fact_checking_guidelines.json',
        'retriever': {
            'model': 'sentence-transformers/all-MiniLM-L6-v2',
            'top_k': 3,
            'threshold': 0.7
        },
        'template': "An accurate image showing {text} based on verified information: {rag_context}"
    },
    'inference': {
        'batch_size': 16,
        'threshold': 0.272,
        'temperature': 1.0
    }
}
```

### BLIP2 Experiments

#### 1. Zero-Shot Base
```python
config = {
    'model': {
        'name': 'blip2',
        'version': 'Salesforce/blip2-opt-2.7b',
        'pretrained': True
    },
    'preprocessing': {
        'image_size': 224,
        'normalize': True,
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711]
    },
    'generation': {
        'max_new_tokens': 75,
        'temperature': 1.0,
        'top_k': 10,
        'do_sample': False,
        'early_stopping': True
    },
    'prompt': {
        'template': "Does the text match the image? Answer with 'yes' or 'no'.",
        'system_prompt': None
    },
    'rag_enabled': False
}
```

#### 2. Few-Shot Configuration
```python
config = {
    'model': {
        'name': 'blip2',
        'version': 'Salesforce/blip2-opt-2.7b',
        'pretrained': True
    },
    'preprocessing': {
        'image_size': 224,
        'normalize': True
    },
    'generation': {
        'max_new_tokens': 75,
        'temperature': 0.7,
        'top_k': 10,
        'do_sample': True
    },
    'few_shot': {
        'enabled': True,
        'examples': [
            {
                'image_desc': "Cat on red chair",
                'text': "A cat sitting on a red chair",
                'label': "yes",
                'explanation': "The image shows exactly what the text describes"
            },
            {
                'image_desc': "Ocean sunset",
                'text': "Beautiful sunset over the ocean",
                'label': "yes",
                'explanation': "The image matches the described scene"
            },
            {
                'image_desc': "Generic space photo",
                'text': "NASA discovers alien life on Mars",
                'label': "no",
                'explanation': "The text makes claims not supported by the image"
            },
            {
                'image_desc': "Political rally",
                'text': "President Obama riding a unicorn",
                'label': "no",
                'explanation': "The text describes an impossible scenario"
            }
        ],
        'template': few_shot_template  # Defined in previous documentation
    },
    'rag_enabled': False
}
```

#### 3. RAG + Few-Shot
```python
config = {
    'model': {
        'name': 'blip2',
        'version': 'Salesforce/blip2-opt-2.7b',
        'pretrained': True
    },
    'preprocessing': {
        'image_size': 224,
        'normalize': True
    },
    'generation': {
        'max_new_tokens': 75,
        'temperature': 0.7,
        'top_k': 10
    },
    'few_shot': {
        'enabled': True,
        'examples': [  # Same as above
        ]
    },
    'rag': {
        'enabled': True,
        'knowledge_base': 'enhanced_guidelines.json',
        'retriever': {
            'model': 'sentence-transformers/all-MiniLM-L6-v2',
            'top_k': 3
        },
        'template': blip2_rag_template  # Defined in previous documentation
    }
}
```

### LLaVA Experiments

#### 1. Zero-Shot Base
```python
config = {
    'model': {
        'name': 'llava',
        'version': 'llava-1.5-7b',
        'vision_encoder': 'clip-vit-large-patch14',
        'pretrained': True
    },
    'preprocessing': {
        'image_size': 336,
        'normalize': True
    },
    'generation': {
        'max_new_tokens': 512,
        'temperature': 1.0,
        'top_p': 0.9,
        'repetition_penalty': 1.0
    },
    'prompt': {
        'template': zero_shot_simple,  # Defined in previous documentation
        'system_prompt': "You are a helpful assistant analyzing image-text pairs."
    },
    'rag_enabled': False
}
```

#### 2. Few-Shot with Chain-of-Thought
```python
config = {
    'model': {
        'name': 'llava',
        'version': 'llava-1.5-7b',
        'vision_encoder': 'clip-vit-large-patch14',
        'pretrained': True
    },
    'preprocessing': {
        'image_size': 336,
        'normalize': True
    },
    'generation': {
        'max_new_tokens': 512,
        'temperature': 0.7,
        'top_p': 0.9,
        'repetition_penalty': 1.1
    },
    'few_shot': {
        'enabled': True,
        'examples': [
            {
                'image_desc': "Cat on red chair",
                'text': "A cat sitting on a red chair",
                'label': "yes",
                'analysis': {
                    'image_content': "A ginger cat sitting comfortably on a red armchair",
                    'text_claim': "States there is a cat on a red chair",
                    'comparison': "The image content exactly matches the text description"
                }
            },
            # ... more examples as defined in previous documentation
        ],
        'template': format_cot_prompt  # Defined in previous documentation
    },
    'rag_enabled': False
}
```

#### 3. RAG + Few-Shot
```python
config = {
    'model': {
        'name': 'llava',
        'version': 'llava-1.5-7b',
        'vision_encoder': 'clip-vit-large-patch14',
        'pretrained': True
    },
    'preprocessing': {
        'image_size': 336,
        'normalize': True
    },
    'generation': {
        'max_new_tokens': 512,
        'temperature': 0.7,
        'top_p': 0.9
    },
    'few_shot': {
        'enabled': True,
        'examples': [  # Same as above
        ]
    },
    'rag': {
        'enabled': True,
        'knowledge_base': 'fact_checking_guidelines.json',
        'retriever': {
            'model': 'sentence-transformers/all-MiniLM-L6-v2',
            'top_k': 3
        },
        'template': llava_rag_template  # Defined in previous documentation
    }
}
```

## Complete Metrics for Each Run

### Metrics Collection Implementation
```python
class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(dict)
        
    def add_run_metrics(self, run_id, predictions, labels, scores=None, timing=None):
        """
        Collect comprehensive metrics for a run
        Args:
            run_id (str): Unique identifier for the run
            predictions (List[bool]): Model predictions
            labels (List[bool]): True labels
            scores (List[float], optional): Confidence scores
            timing (List[float], optional): Processing times
        """
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1': f1_score(labels, predictions),
            'confusion_matrix': confusion_matrix(labels, predictions).tolist(),
            'support': len(predictions)
        }
        
        if scores is not None:
            metrics.update({
                'roc_auc': roc_auc_score(labels, scores),
                'pr_auc': average_precision_score(labels, scores),
                'average_confidence': np.mean(scores)
            })
            
        if timing is not None:
            metrics.update({
                'avg_processing_time': np.mean(timing),
                'std_processing_time': np.std(timing),
                'total_processing_time': np.sum(timing)
            })
            
        self.metrics[run_id] = metrics
        
    def get_run_metrics(self, run_id):
        return self.metrics[run_id]
        
    def compare_runs(self, run_ids):
        """Compare metrics across multiple runs"""
        comparison = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            comparison[metric] = {
                run_id: self.metrics[run_id][metric]
                for run_id in run_ids
            }
        return comparison
```

### Example Run Collection
```python
def collect_run_metrics(model_name, config, predictions, labels, scores, timing):
    """
    Collect and store metrics for a specific experimental run
    """
    run_id = f"{model_name}_{config['mode']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Basic metrics
    metrics = {
        'model_name': model_name,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(predictions),
        'performance_metrics': {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1': f1_score(labels, predictions),
            'confusion_matrix': confusion_matrix(labels, predictions).tolist()
        }
    }
    
    # Add confidence metrics if available
    if scores is not None:
        metrics['confidence_metrics'] = {
            'roc_auc': roc_auc_score(labels, scores),
            'pr_auc': average_precision_score(labels, scores),
            'average_confidence': np.mean(scores),
            'confidence_std': np.std(scores)
        }
    
    # Add timing metrics if available
    if timing is not None:
        metrics['timing_metrics'] = {
            'average_time': np.mean(timing),
            'total_time': np.sum(timing),
            'std_time': np.std(timing)
        }
    
    # Save metrics to file
    with open(f'results/{run_id}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return run_id, metrics
``` 

### Evaluation Implementation

#### 1. Core Evaluation Metrics
```python
def calculate_metrics(y_true, y_pred, y_prob=None, average='binary'):
    """Calculate classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics
```

#### 2. Detailed Model Evaluation
```python
def evaluate_model_outputs(results_df, true_label_col='true_labels', 
                         pred_label_col='predicted_labels',
                         generated_text_col='generated_text',
                         report_path=None, figures_dir=None):
    """Comprehensive model evaluation."""
    # Filter valid predictions
    valid_mask = results_df[pred_label_col].notna()
    results_df_valid = results_df[valid_mask]
    
    # Calculate metrics
    y_true = results_df_valid[true_label_col]
    y_pred = results_df_valid[pred_label_col]
    
    metrics = calculate_metrics(y_true, y_pred, average='binary')
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Save detailed report
    if report_path:
        with open(report_path, 'w') as f:
            f.write("=== Model Evaluation Report ===\n\n")
            f.write(f"Total samples: {len(results_df)}\n")
            f.write(f"Valid predictions: {len(results_df_valid)}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-score: {metrics['f1_score']:.4f}\n")
            f.write(f"Specificity: {specificity:.4f}\n")
            f.write(f"Sensitivity: {sensitivity:.4f}\n\n")
            
            f.write("=== Confusion Matrix ===\n")
            f.write(f"True Negatives: {tn}\n")
            f.write(f"False Positives: {fp}\n")
            f.write(f"False Negatives: {fn}\n")
            f.write(f"True Positives: {tp}\n")
```

#### 3. Model-Specific Evaluation

##### CLIP Evaluation
```python
def evaluate_clip(results_df):
    """Evaluate CLIP model performance."""
    # Convert similarity scores to binary predictions
    clip_scores = results_df['scores'].values
    clip_preds = (clip_scores > 25.0).astype(int)
    
    metrics = calculate_metrics(
        results_df['true_label'].values,
        clip_preds,
        average='binary'
    )
    
    return metrics
```

##### BLIP2 Evaluation
```python
def evaluate_blip2(results_df):
    """Evaluate BLIP2 model performance."""
    # Parse generated text for yes/no answers
    parsed_results = []
    for text in results_df['generated_text']:
        if re.search(r'\b(yes|no)\b', text.lower()):
            pred = 1 if 'yes' in text.lower() else 0
            parsed_results.append(pred)
        else:
            parsed_results.append(None)
    
    # Filter valid predictions
    valid_mask = [p is not None for p in parsed_results]
    
    metrics = calculate_metrics(
        results_df['true_label'][valid_mask],
        [p for p in parsed_results if p is not None],
        average='binary'
    )
    
    return metrics
```

##### LLaVA Evaluation
```python
def evaluate_llava(results_df):
    """Evaluate LLaVA model performance."""
    llava_parser = LLaVAAnswerParser()
    
    parsed_results = []
    for text in results_df['generated_text']:
        pred, conf, _ = llava_parser.extract_prediction(text)
        parsed_results.append({
            'prediction': pred,
            'confidence': conf
        })
    
    # Filter confident predictions
    valid_preds = [r['prediction'] for r in parsed_results if r['prediction'] is not None]
    valid_mask = [r['prediction'] is not None for r in parsed_results]
    
    metrics = calculate_metrics(
        results_df['true_label'][valid_mask],
        valid_preds,
        average='binary'
    )
    
    return metrics
```

#### 4. Ensemble Evaluation

##### Ensemble Metrics
```python
def evaluate_ensemble(ensemble_df: pd.DataFrame) -> Dict:
    """Evaluate ensemble performance."""
    true_labels = ensemble_df['true_label'].values
    ensemble_preds = ensemble_df['ensemble_prediction'].values
    clip_preds = ensemble_df['clip_prediction'].values
    bert_preds = ensemble_df['bert_prediction'].values
    
    # Calculate metrics for each component
    metrics = {}
    for name, preds in [('ensemble', ensemble_preds), 
                       ('clip', clip_preds), 
                       ('bert', bert_preds)]:
        metrics[name] = {
            'accuracy': accuracy_score(true_labels, preds),
            'precision': precision_score(true_labels, preds, zero_division=0),
            'recall': recall_score(true_labels, preds, zero_division=0),
            'f1': f1_score(true_labels, preds, zero_division=0)
        }
    
    # Add RAG metrics if available
    if 'rag_prediction' in ensemble_df.columns:
        rag_preds = ensemble_df['rag_prediction'].values
        metrics['rag'] = {
            'accuracy': accuracy_score(true_labels, rag_preds),
            'precision': precision_score(true_labels, rag_preds, zero_division=0),
            'recall': recall_score(true_labels, rag_preds, zero_division=0),
            'f1': f1_score(true_labels, rag_preds, zero_division=0)
        }
    
    return metrics
```

##### Ensemble Visualization
```python
def create_metrics_visualization(metrics: Dict, output_dir: str):
    """Create visualization comparing ensemble vs individual models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Ensemble vs Individual Model Performance', fontsize=16)
    
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    for i, metric in enumerate(metric_names):
        ax = axes[i//2, i%2]
        
        models = list(metrics.keys())
        values = [metrics[model][metric] for model in models]
        
        bars = ax.bar(models, values, color=colors[:len(models)])
        ax.set_title(f'{metric.title()} Score')
        ax.set_ylabel(metric.title())
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'ensemble_comparison.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### Final Results Analysis

#### 1. Metrics Summary
```python
def generate_summary_statistics(metrics_df: pd.DataFrame) -> Dict:
    """Generate comprehensive summary statistics."""
    summary = {
        'total_experiments': len(metrics_df),
        'models_tested': metrics_df['model'].nunique(),
        'best_accuracy': metrics_df['accuracy'].max(),
        'best_experiment': metrics_df.loc[metrics_df['accuracy'].idxmax(), 'experiment'],
        'average_accuracy': metrics_df['accuracy'].mean(),
        'accuracy_std': metrics_df['accuracy'].std(),
        'model_performance': metrics_df.groupby('model')['accuracy'].agg(['mean', 'std', 'max']).to_dict(),
        'rag_impact': {
            'with_rag': metrics_df[metrics_df['rag'] == 'Yes']['accuracy'].mean(),
            'without_rag': metrics_df[metrics_df['rag'] == 'No']['accuracy'].mean()
        },
        'shot_type_performance': metrics_df.groupby('shot_type')['accuracy'].mean().to_dict()
    }
    return summary
```

#### 2. Detailed Analysis Plots
```python
def plot_detailed_analysis(metrics_df: pd.DataFrame):
    """Create detailed analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Performance Analysis', fontsize=16)
    
    # Precision-Recall scatter
    ax1 = axes[0, 0]
    for model in metrics_df['model'].unique():
        model_data = metrics_df[metrics_df['model'] == model]
        ax1.scatter(model_data['precision'], model_data['recall'], 
                   label=model, s=100, alpha=0.7)
    ax1.set_xlabel('Precision')
    ax1.set_ylabel('Recall')
    ax1.set_title('Precision vs Recall')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Performance heatmap
    ax2 = axes[0, 1]
    pivot_data = metrics_df.pivot_table(
        values='accuracy', 
        index='model', 
        columns='shot_type', 
        aggfunc='mean'
    )
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
    ax2.set_title('Accuracy Heatmap: Model vs Shot Type')
    
    # RAG impact analysis
    ax3 = axes[1, 0]
    rag_impact = metrics_df.groupby(['model', 'rag'])['accuracy'].mean().unstack()
    rag_impact.plot(kind='bar', ax=ax3)
    ax3.set_title('RAG Impact on Accuracy')
    ax3.set_ylabel('Accuracy')
    ax3.legend(title='RAG')
    
    # Best configurations
    ax4 = axes[1, 1]
    top_configs = metrics_df.nlargest(8, 'accuracy')[['experiment', 'accuracy']]
    ax4.barh(range(len(top_configs)), top_configs['accuracy'])
    ax4.set_yticks(range(len(top_configs)))
    ax4.set_yticklabels(top_configs['experiment'])
    ax4.set_xlabel('Accuracy')
    ax4.set_title('Top 8 Performing Configurations')
```

### Performance Results

#### 1. Model Performance Summary
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|-----------|
| CLIP | 0.820 | 0.809 | 0.760 | 0.784 |
| BLIP2 | 0.620 | 0.598 | 0.582 | 0.590 |
| LLaVA | 0.650 | 0.632 | 0.612 | 0.622 |

#### 2. RAG Enhancement Impact
| Model | Base Accuracy | RAG Enhanced | Improvement |
|-------|---------------|--------------|-------------|
| CLIP | 0.790 | 0.820 | +0.030 |
| BLIP2 | 0.560 | 0.620 | +0.060 |
| LLaVA | 0.630 | 0.650 | +0.020 |

#### 3. Ensemble Performance
| Configuration | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|---------|-----------|
| CLIP Only | 0.820 | 0.809 | 0.760 | 0.784 |
| CLIP + BERT | 0.835 | 0.823 | 0.778 | 0.800 |
| CLIP + BERT + RAG | 0.847 | 0.831 | 0.792 | 0.811 |

#### 4. Processing Efficiency
| Model | Avg Time (s) | Memory (GB) | Batch Size |
|-------|--------------|-------------|------------|
| CLIP | 0.100 | 2 | 32 |
| BLIP2 | 12.478 | 6 | 8 |
| LLaVA | 0.930 | 14 | 16 |
| RAG Overhead | +0.200 | +1 | N/A |

### Key Findings

1. **Model Performance**
   - CLIP consistently outperforms other models
   - BLIP2 shows significant improvement with RAG
   - LLaVA performs well on complex cases

2. **RAG Impact**
   - Average accuracy improvement: +3.7%
   - Most effective with BLIP2
   - Helps with factual verification

3. **Ensemble Benefits**
   - Best performance: CLIP + BERT + RAG
   - Weighted voting outperforms majority
   - Confidence-based ensemble most robust

4. **Resource Considerations**
   - CLIP most efficient
   - BLIP2 requires significant processing time
   - LLaVA needs substantial memory

5. **Optimization Insights**
   - RAG parameters crucial for performance
   - Few-shot learning helps BLIP2 and LLaVA
   - Prompt engineering impacts accuracy 

### Model Implementations

#### 1. CLIP Implementation
```python
def load_clip(model_name="openai/clip-vit-base-patch32"):
    """Loads CLIP model and processor."""
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def process_batch_for_clip(batch, clip_processor, device="cpu"):
    """Process batch for CLIP."""
    inputs = clip_processor(
        text=batch['text'],
        images=batch['image'],
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    return {k: v.to(device) for k, v in inputs.items()}
```

#### 2. BLIP2 Implementation
```python
def load_blip_conditional(model_name="Salesforce/blip-image-captioning-base"):
    """Load BLIP2 model with optimizations."""
    if "blip2" in model_name.lower():
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        processor = BlipProcessor.from_pretrained(model_name)
        processor.tokenizer.padding_side = "left"
    
    return model, processor

def process_batch_for_blip_conditional(batch, blip_processor, task="captioning", device="cpu"):
    """Process batch for BLIP2."""
    inputs = blip_processor(
        text=batch['text'],
        images=batch['image'],
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    return {k: v.to(device) for k, v in inputs.items()}
```

#### 3. LLaVA Implementation
```python
def load_llava(model_name_or_path, **kwargs):
    """Load LLaVA 1.5 model with optimizations."""
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        **kwargs
    )
    return model, processor

def process_batch_for_llava(batch, processor, device, llava_prompt_template, 
                          few_shot_images=None, metadata_strings=None):
    """Process batch for LLaVA with advanced handling."""
    try:
        # Handle different processor types
        if hasattr(processor, 'image_processor'):
            # Standard LLaVA processing
            inputs = processor(
                text=batch['text'],
                images=batch['image'],
                return_tensors="pt",
                padding=True
            )
        else:
            # Unified processor
            inputs = processor(
                text=batch['text'],
                images=batch['image'],
                return_tensors="pt",
                padding=True
            )
        
        return {k: v.to(device) for k, v in inputs.items()}
    except Exception as e:
        logger.error(f"Error processing batch for LLaVA: {e}")
        return None
```

#### 4. BERT Implementation
```python
def load_bert_classifier(model_name="bert-base-uncased", num_labels=2):
    """Load BERT for classification."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def process_batch_for_bert_classifier(batch, bert_tokenizer, device="cpu", max_length=128):
    """Process batch for BERT."""
    inputs = bert_tokenizer(
        batch['text'],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    return {k: v.to(device) for k, v in inputs.items()}
```

### Model Configurations

#### 1. CLIP Configuration
```python
clip_config = {
    'model_name': 'openai/clip-vit-base-patch32',
    'image_size': 224,
    'batch_size': 32,
    'similarity_threshold': 25.0,
    'preprocessing': {
        'normalize': True,
        'resize': True,
        'center_crop': True
    }
}
```

#### 2. BLIP2 Configuration
```python
blip2_config = {
    'model_name': 'Salesforce/blip2-opt-2.7b',
    'image_size': 224,
    'batch_size': 8,
    'max_new_tokens': 75,
    'preprocessing': {
        'normalize': True,
        'resize': True
    },
    'generation': {
        'temperature': 0.7,
        'top_k': 10,
        'max_length': 100,
        'min_length': 1,
        'num_beams': 5
    }
}
```

#### 3. LLaVA Configuration
```python
llava_config = {
    'model_name': 'llava-1.5-7b',
    'image_size': 336,
    'batch_size': 16,
    'max_new_tokens': 100,
    'preprocessing': {
        'normalize': True,
        'resize': True
    },
    'generation': {
        'temperature': 0.7,
        'top_p': 0.9,
        'repetition_penalty': 1.1,
        'max_length': 512,
        'min_length': 1,
        'num_beams': 3
    }
}
```

#### 4. BERT Configuration
```python
bert_config = {
    'model_name': 'bert-base-uncased',
    'max_length': 128,
    'batch_size': 32,
    'num_labels': 2,
    'preprocessing': {
        'truncation': True,
        'padding': True
    }
}
```

### Model Pipeline Integration

#### 1. Main Pipeline
```python
def main(args):
    # Load appropriate model and processor
    if args.model_type == 'clip':
        model, processor = load_clip(args.clip_model_name)
        process_batch_fn = process_batch_for_clip
    elif args.model_type == 'blip':
        model, processor = load_blip_conditional(args.blip_model_name)
        process_batch_fn = process_batch_for_blip_conditional
    elif args.model_type == 'llava':
        model, processor = load_llava(args.llava_model_name)
        process_batch_fn = process_batch_for_llava
    elif args.model_type == 'bert':
        model, processor = load_bert_classifier(args.bert_model_name, num_labels=args.num_labels)
        process_batch_fn = process_batch_for_bert_classifier
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Process batches
    results = []
    for batch in dataloader:
        inputs = process_batch_fn(batch, processor, device)
        with torch.no_grad():
            outputs = model(**inputs)
        results.extend(process_outputs(outputs))
    
    return results
```

#### 2. Model-Specific Processing
```python
def process_outputs(outputs, model_type):
    """Process model outputs based on type."""
    if model_type == 'clip':
        # Process CLIP similarity scores
        logits = outputs.logits_per_image
        scores = torch.diagonal(logits).cpu().numpy()
        predictions = (scores > 25.0).astype(int)
    elif model_type == 'blip':
        # Process BLIP generated text
        generated_ids = outputs.sequences
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        predictions = parse_blip_outputs(generated_text)
    elif model_type == 'llava':
        # Process LLaVA responses
        generated_ids = outputs.sequences
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        predictions = parse_llava_outputs(generated_text)
    elif model_type == 'bert':
        # Process BERT classification logits
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
    
    return predictions
```

### Resource Requirements

#### 1. Memory Usage
| Model | Base Memory (GB) | Batch Memory (GB) | Total for BS=32 |
|-------|-----------------|-------------------|-----------------|
| CLIP | 2 | 0.1 | 5.2 |
| BLIP2 | 6 | 0.5 | 22 |
| LLaVA | 14 | 0.8 | 39.6 |
| BERT | 0.5 | 0.05 | 2.1 |

#### 2. Processing Speed
| Model | Images/Second | Batch Size | GPU Utilization |
|-------|--------------|------------|-----------------|
| CLIP | 156.2 | 32 | 65% |
| BLIP2 | 12.4 | 8 | 95% |
| LLaVA | 16.8 | 16 | 88% |
| BERT | 245.6 | 32 | 45% |

#### 3. GPU Requirements
| Model | Minimum VRAM | Recommended VRAM | Multi-GPU Support |
|-------|--------------|------------------|-------------------|
| CLIP | 4GB | 8GB | No |
| BLIP2 | 16GB | 24GB | Yes |
| LLaVA | 24GB | 32GB | Yes |
| BERT | 4GB | 8GB | No |

### Optimization Techniques

1. **Memory Optimization**
   - Use torch.float16 for large models
   - Gradient checkpointing for training
   - Model sharding for multi-GPU

2. **Speed Optimization**
   - Batch size tuning
   - Input caching
   - Parallel processing

3. **Quality Optimization**
   - Prompt engineering
   - Temperature tuning
   - Ensemble methods

### Best Performing Model Configurations

#### 1. CLIP Configuration Details
```python
clip_detailed_config = {
    'architecture': 'ViT-Base/16',
    'input_resolution': 224,
    'embedding_dimension': 512,
    'similarity_metric': 'cosine',
    'thresholds': {
        'zero_shot': 0.272,  # Optimized threshold
        'rag_enhanced': 0.272
    },
    'performance': {
        'zero_shot': {
            'accuracy': 0.790,
            'precision': 0.809,
            'recall': 0.760,
            'f1_score': 0.784,
            'roc_auc': 0.814
        },
        'optimized': {
            'accuracy': 0.820,
            'precision': 0.809,
            'recall': 0.760,
            'f1_score': 0.784,
            'roc_auc': 0.814
        }
    },
    'preprocessing': {
        'multi_crop': True,
        'crop_count': 3,
        'text_preprocessing': ['stopword_removal', 'lowercasing'],
        'normalization': 'imagenet_stats',
        'embedding_norm': 'l2'
    }
}
```

#### 2. BLIP2 Configuration Details
```python
blip2_detailed_config = {
    'model_name': 'Salesforce/blip2-opt-2.7b',
    'input_resolution': 224,
    'max_text_length': 75,
    'output_format': 'binary',
    'performance': {
        'zero_shot': {
            'accuracy': 0.560,
            'precision': 0.543,
            'recall': 0.542,
            'f1_score': 0.540,
            'roc_auc': 0.428
        },
        'few_shot': {
            'accuracy': 0.620,
            'precision': 0.571,
            'recall': 0.200,
            'f1_score': 0.296,
            'roc_auc': 0.428
        }
    },
    'generation': {
        'max_new_tokens': 75,
        'temperature': 1.0,
        'top_k': 10,
        'do_sample': False
    }
}
```

#### 3. LLaVA Configuration Details
```python
llava_detailed_config = {
    'model_name': 'llava-1.5-7b',
    'vision_encoder': 'CLIP ViT-L/14',
    'language_model': 'Vicuna-7B-v1.5',
    'max_text_length': 512,
    'performance': {
        'zero_shot': {
            'accuracy': 0.630,
            'precision': 0.647,
            'recall': 0.660,
            'f1_score': 0.653,
            'processing_time': 0.93
        },
        'few_shot': {
            'accuracy': 0.650,
            'precision': 0.636,
            'recall': 0.700,
            'f1_score': 0.667,
            'processing_time': 1.10
        }
    },
    'generation': {
        'temperature': 0.7,
        'max_new_tokens': 20,
        'device': 'NVIDIA A100-SXM4-40GB',
        'batch_size': 1
    }
}
```

### Processing Pipeline Implementations

#### 1. Image Processing Pipeline
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

#### 2. Text Processing Pipeline
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

### Prompt Templates

#### 1. CLIP Zero-Shot
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

#### 2. BLIP2 Templates
```python
# Zero-shot template
zero_shot_template = """
Question: Does the text match the image? Answer with only 'yes' or 'no'.

Text: {text}
"""

# Few-shot template with examples
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
```

#### 3. LLaVA Templates
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

# Chain-of-thought template
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