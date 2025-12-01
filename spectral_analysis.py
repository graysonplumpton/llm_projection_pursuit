import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

model_path = "/model-weights/Llama-3.2-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",    
    attn_implementation="flash_attention_2"
)

layer_idx = 14
W = model.model.layers[layer_idx].mlp.down_proj.weight.detach().float().cpu().numpy()

print(f"Weight matrix shape: {W.shape}")

# Compute W^T W (or WW^T if W is wide)
# Use the smaller dimension for efficiency
if W.shape[0] < W.shape[1]:
    M = W @ W.T
else:
    M = W.T @ W


eigenvalues = np.linalg.eigvalsh(M)

# Compute lambda for Marchenko-Pastur comparison
lambda_mp = min(W.shape) / max(W.shape)
print(f"Lambda for MP comparison: {lambda_mp:.3f}")

# Plot the spectrum as a histogram
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 0.5,
    'axes.edgecolor': 'black',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)  # High DPI for resolution

# Histogram with styling to match your PDF (no edge color, solid blue bars)
ax.hist(eigenvalues, bins=60, color='#7fadd3', edgecolor='#1F77B4', alpha=1.0, linewidth=0.3)            
ax.set_xlabel("Eigenvalue", fontsize=12)
ax.set_ylabel("Count", fontsize=12)

# Clean up the axes to match the minimal style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Log to wandb with high resolution
if log_to_wandb and wandb.run is not None:
    # Save to a buffer at high resolution
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    pil_image = Image.open(buf)
    
    wandb.log({
        "spectrum_plot": wandb.Image(pil_image),
        "eigenvalues": wandb.Histogram(eigenvalues),
        "max_eigenvalue": eigenvalues.max(),
        "min_eigenvalue": eigenvalues.min(),
    }, step=step)
    buf.close()

plt.close(fig)
