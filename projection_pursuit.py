import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score
import wandb

def projection_pursuit(data, labels, n_projections=2000):
    """
    Projects high-dimensional data to random 2D subspaces and finds 
    the projections that maximize the Calinski-Harabasz score.
    
    Args:
        data (torch.Tensor): Input data of shape (n_samples, n_features)
        labels (torch.Tensor or array): Ground truth labels or cluster assignments
        n_projections (int): Number of random projections to try
        
    Returns:
        list: A list of the top 5 dictionaries, each containing:
              {'score': float, 'data_2d': torch.Tensor, 'projection_matrix': torch.Tensor}
    """
    
    n_features = data.shape[1]
    
    # We will store results here instead of a dictionary with tensor keys
    results = []

    # Ensure labels are in a format sklearn likes (numpy, on cpu)
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = labels

    print(f"Starting {n_projections} iterations of projection pursuit...")

    for i in range(n_projections):
        # 1. Generate random matrix (n_features, 2)
        random_matrix = torch.randn(n_features, 2, device=data.device)
        
        # 2. QR Decomposition to get an orthogonal basis (subspace)
        # We only need Q (the orthogonal columns)
        proj_matrix, _ = torch.linalg.qr(random_matrix)
        
        # 3. Project data: (N, D) @ (D, 2) -> (N, 2)
        data_2d = data @ proj_matrix
        
        # 4. Compute Score
        # We must detach and move to CPU for sklearn
        data_2d_np = data_2d.detach().cpu().numpy()
        
        try:
            score = calinski_harabasz_score(data_2d_np, labels_np)
        except ValueError:
            # Handle edge cases (e.g., if a projection collapses a class to a single point)
            score = 0.0

        # 5. Store result
        results.append({
            "score": score,
            "data_2d": data_2d,
            "projection_matrix": proj_matrix
        })

    # 6. Sort by score (descending) and keep top 5
    top_5_results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
    
    print("Projection pursuit complete.")
    return top_5_results

def plot_top_projections(top_results, labels, class_names=None, log_to_wandb=False):
    """
    Visualizes the top 5 projections found by the pursuit algorithm.
    If log_to_wandb is True, logs the figure and a summary table to the active wandb run.
    """
    if not top_results:
        print("No results to plot.")
        return

    # Handle case where labels is a tensor
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = labels

    unique_labels = np.unique(labels_np)
    
    # Handle class_names normalization
    if class_names is None:
        label_map = {l: f"Class {l}" for l in unique_labels}
    elif isinstance(class_names, list):
        label_map = {i: name for i, name in enumerate(class_names)}
    elif isinstance(class_names, dict):
        label_map = class_names
    else:
        label_map = {l: str(l) for l in unique_labels}

    # --- 1. Create the Main Composite Plot ---
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    if len(top_results) < 5:
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

    for i, result in enumerate(top_results):
        ax = axes[i]
        points = result['data_2d'].cpu().numpy()
        score = result['score']
        
        # Plot each class separately to generate a legend
        for label_val in unique_labels:
            mask = labels_np == label_val
            ax.scatter(
                points[mask, 0], 
                points[mask, 1], 
                label=label_map.get(label_val, str(label_val)),
                s=20, 
                alpha=0.7
            )
            
        ax.set_title(f"Rank {i+1}\nCH Score: {score:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add legend to the first plot
        if i == 0:
            ax.legend(loc='upper right', fontsize='small')
        
    plt.tight_layout()
    
    # --- 2. WandB Logging ---
    if log_to_wandb and wandb.run is not None:
        print("Logging results to wandb...")
        
        # Log the main overview figure
        wandb.log({"Top 5 Projections (Overview)": wandb.Image(fig)})
        
        # Log a detailed Table with individual plots
        table = wandb.Table(columns=["Rank", "CH Score", "Plot"])
        
        for i, result in enumerate(top_results):
            fig_single, ax_single = plt.subplots(figsize=(6, 5))
            points = result['data_2d'].cpu().numpy()
            
            for label_val in unique_labels:
                mask = labels_np == label_val
                ax_single.scatter(
                    points[mask, 0], 
                    points[mask, 1], 
                    label=label_map.get(label_val, str(label_val)),
                    s=20, 
                    alpha=0.7
                )
            
            ax_single.set_title(f"Score: {result['score']:.2f}")
            ax_single.legend()
            ax_single.axis('off')
            plt.tight_layout()
            
            table.add_data(i+1, result['score'], wandb.Image(fig_single))
            plt.close(fig_single)
            
        wandb.log({"Top Projections Table": table})

    plt.show()
