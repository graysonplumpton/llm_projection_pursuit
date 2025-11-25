import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import your custom tools
try:
    from projection_pursuit import projection_pursuit, plot_top_projections
except ImportError:
    print("Warning: Could not import projection_pursuit. Ensure the file is in the same directory.")

def generate_templates(n_templates=100):
    """
    Generates a list of sentence templates containing the {color} placeholder.
    """
    # Basic lists to construct varied sentences
    subjects = [
        "car", "house", "flower", "bird", "dress", "sky", "book", "flag", 
        "apple", "pen", "shoe", "bag", "chair", "table", "phone", "wall", 
        "cup", "plate", "hat", "scarf", "bike", "door", "couch", "rug",
        "toy", "balloon", "kite", "ribbon", "gem", "eye", "hair", "fish",
        "frog", "snake", "leaf", "tree", "sign", "light", "paint", "ink"
    ]
    
    structures = [
        "The {subj} was painted a bright {color}.",
        "I couldn't believe the {subj} was actually {color}.",
        "She picked out a lovely {color} {subj} for the party.",
        "Look at that {color} {subj} over there.",
        "My favorite object is the {color} {subj}.",
        "He decided to buy the {color} {subj} instead of the black one.",
        "The artist used a lot of {color} on the {subj}.",
        "Why is the {subj} turning {color}?",
        "A small {color} {subj} sat on the desk.",
        "We need to find a matching {color} {subj}.",
        "The specific shade of the {subj} was {color}.",
        "It was a deep, rich {color} {subj}.",
        "They say a {color} {subj} brings good luck.",
        "The {subj} faded to a dull {color} over time.",
        "I've never seen a {color} {subj} before."
    ]
    
    templates = []
    # Generate combinations until we have enough
    count = 0
    for s in structures:
        for subj in subjects:
            if count >= n_templates:
                break
            templates.append(s.format(subj=subj, color="{color}"))
            count += 1
            
    # If we ran out of combinations (unlikely with this math), cycle
    while len(templates) < n_templates:
        templates.append(templates[count % len(templates)])
        count += 1
        
    return templates[:n_templates]

def get_color_embeddings(model, tokenizer, templates, colors, device):
    """
    Extracts the contextual embedding of the color token from each sentence.
    """
    embeddings = []
    labels = []
    
    print(f"Extracting embeddings for {len(templates) * len(colors)} sentences...")
    
    model.eval()
    
    for template in tqdm(templates):
        for color_idx, color_name in enumerate(colors):
            # Construct sentence
            text = template.format(color=color_name)
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt").to(device)
            input_ids = inputs.input_ids[0]
            
            # --- Find the position of the color token ---
            # Strategy: Tokenize the color word alone (with a leading space) 
            # and find that sequence in the sentence.
            # Note: " red" might be different from "red". Templates usually provide a space.
            
            # We try finding the token corresponding to the color.
            # A robust way for Llama-3 is to use character offsets or simple string matching 
            # on the decoded tokens, but here is a heuristic that works well for single words:
            
            # 1. Decode each token and strip spaces
            tokens_str = [tokenizer.decode([t]).strip() for t in input_ids]
            
            # 2. Find index where the token string matches the color name
            # Note: This finds the *first* occurrence. 
            target_pos = -1
            for i, t_str in enumerate(tokens_str):
                # Simple check (case insensitive mostly, but color names are lower)
                if color_name.lower() in t_str.lower():
                    target_pos = i
                    break
            
            if target_pos == -1:
                # Fallback: if tokenization split the color (e.g. "in" + "digo")
                # We skip or take the last token. For now, let's just warn and skip.
                # (Llama 3 usually handles basic colors as single tokens)
                print(f"Warning: Could not find color '{color_name}' in tokens: {tokens_str}")
                continue

            # --- Forward Pass ---
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            # Extract embedding from the last layer at the found position
            # Shape: (batch, seq_len, hidden_dim) -> (hidden_dim,)
            # We use the last hidden state (layer -1)
            token_embedding = outputs.hidden_states[-1][0, target_pos, :].cpu()
            
            embeddings.append(token_embedding)
            labels.append(color_idx)

    return torch.stack(embeddings), torch.tensor(labels)

def run_experiment():
    wandb.init(project="llama-color-geometry")

    # --- 1. Model Setup ---
    print("Loading model...")
    model_path = "/model-weights/Llama-3.2-3B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",    
        attn_implementation="flash_attention_2"
    )
    
    print(f"Model loaded on {device}")

    # --- 2. Data Generation ---
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    templates = generate_templates(n_templates=100)
    
    # --- 3. Extract Embeddings (700 points) ---
    embeddings, labels = get_color_embeddings(model, tokenizer, templates, colors, device)
    print(f"Extracted embeddings shape: {embeddings.shape}") # Should be (700, 3072) for 3B

    # --- 4. Dimensionality Reduction (PCA -> 50) ---
    print("Normalizing and Centering data...")
    # Center (subtract mean)
    embeddings = embeddings - embeddings.mean(dim=0)
    # L2 Normalize (project to sphere)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    print("Running PCA reduction to 50 dimensions...")
    pca = PCA(n_components=50)
    # Convert bfloat16 to float32 for sklearn
    embeddings_np = embeddings.float().numpy()
    data_pca_np = pca.fit_transform(embeddings_np)
    data_pca = torch.tensor(data_pca_np)
    
    print(f"PCA shape: {data_pca.shape}")

    # --- 5. Projection Pursuit ---
    # We use the 'colors' list as the class names for the plot
    print("Running Projection Pursuit...")
    top_results = projection_pursuit(data_pca, labels, n_projections=2000)

    # --- 6. Visualization ---
    print("Plotting results...")
    # Map indices to color names
    # Optional: We could map specific colors to the plot points (e.g. 'red' points are actually red)
    # but the standard function maps by index. The labels will be correct though.
    plot_top_projections(
        top_results, 
        labels, 
        class_names=colors, 
        log_to_wandb=True
    )
    
    wandb.finish()

if __name__ == "__main__":
    run_experiment()
