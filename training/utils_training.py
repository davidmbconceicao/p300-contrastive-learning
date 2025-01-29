import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def plot_tsne(embeddings, true_labels, projections=None):
    embeddings = embeddings.cpu().numpy()
    true_labels = true_labels.squeeze().cpu().numpy()
    # Reduce dimensions with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Determine the number of subplots
    if projections is not None:
        projections = projections.cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        projections_2d = tsne.fit_transform(projections)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))

    # Plot using true labels
    scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=true_labels, cmap='viridis', s=5)
    ax1.set_title("t-SNE Visualization of Embeddings with True Labels")
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")

    # Add colorbar for true labels
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label("True Label")

    # Plot using clusters if provided
    if projections is not None:
        scatter2 = ax2.scatter(projections_2d[:, 0], projections_2d[:, 1], c=true_labels, cmap='viridis', s=5)
        ax2.set_title("t-SNE Visualization of Projections with True Labels")
        ax2.set_xlabel("t-SNE Dimension 1")
        ax2.set_ylabel("t-SNE Dimension 2")

        # Add colorbar for clusters
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label("True Label")

    plt.tight_layout()
    # Return the figure
    return fig

# def plot_tsne(normalized_embeddings, true_labels, clusters=None):
#     # Reduce dimensions with t-SNE
#     tsne = TSNE(n_components=2, random_state=42)
#     embeddings_2d = tsne.fit_transform(normalized_embeddings)

#     # Create a figure for true labels
#     if clusters: 
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
#     else: 
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
#     cm = Colormap('colorbrewer:PuOr_4') 
#     # Plot using true labels
#     scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=true_labels, cmap='viridis', s=5)
    
#     ax1.set_title("t-SNE Visualization of Embeddings with True Labels")
#     ax1.set_xlabel("t-SNE Dimension 1")
#     ax1.set_ylabel("t-SNE Dimension 2")
    
#     # Add colorbar for true labels
#     cbar1 = plt.colorbar(scatter1, ax=ax1)
#     cbar1.set_label("True Label")

#     if clusters: 
#         # Plot using clusters
#         scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='viridis', s=5)
#         ax2.set_title("t-SNE Visualization of Embeddings with Clustering (KNN)")
#         ax2.set_xlabel("t-SNE Dimension 1")
#         ax2.set_ylabel("t-SNE Dimension 2")
        
#         # Add colorbar for clusters
#         cbar2 = plt.colorbar(scatter2, ax=ax2)
#         cbar2.set_label("Cluster ID")

#     plt.tight_layout()
#     # Return the figure
#     return fig
    

def plot_views(views):
    fig, axes = plt.subplots(1, len(views), figsize=(10, 8))
    for i in range(len(views)):
        axes[i].imshow(torch.squeeze(views[i]), cmap='inferno', interpolation='none')
        axes[i].axis('off')  
    plt.tight_layout() 
    plt.close(fig)
    return fig

def plot_epoch(epoch):
    plt.imshow(epoch, cmap='inferno', interpolation='none')
    plt.show()
    
def plot_results_across_trials(results:list, n_trials):
    
    # Plotting the data
    fig, ax = plt.subplots()

    
    ax.plot(n_trials, results, marker='o')

    # Adding labels and title
    ax.set_xlabel('Number of Trials')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 105)
    ax.set_title('Accuracy Across Number of Trials')
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_confusion_matrix(confusion_mat, metrics):
    """
    Plots a confusion matrix along with metrics below it.
    
    Args:
        confusion_mat (np.ndarray): The confusion matrix to plot.
        metrics (dict): A dictionary containing metrics such as 
                        accuracy, f1_score, recall, precision, and auc.
        
    Returns:
        fig: The matplotlib figure containing the confusion matrix plot with metrics.
    """
    # Create a new figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Plot the confusion matrix
    confusion_mat = confusion_mat.cpu()
    cax = ax1.matshow(confusion_mat, cmap=plt.cm.Blues)

    # Add a color bar
    plt.colorbar(cax, ax=ax1)

    # Set axis labels
    ax1.set_xlabel('Predicted label')
    ax1.set_ylabel('True label')
    ax1.set_title('Confusion Matrix')

    # Set the ticks and tick labels
    tick_marks = np.arange(len(['Non-P300', 'P300']))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(['Non-P300', 'P300'])
    ax1.set_yticklabels(['Non-P300', 'P300'])

    # Annotate each cell in the confusion matrix
    thresh = confusion_mat.max() / 2.  # Threshold for color
    for i, j in np.ndindex(confusion_mat.shape):
        ax1.text(j, i, format(confusion_mat[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if confusion_mat[i, j] > thresh else "black")

    # Plot metrics below the confusion matrix
    ax2.axis('off')  # Hide the axes for the metrics
    metrics_text = "\n".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
    ax2.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()  # Adjust layout for better spacing
    plt.close(fig)
    return fig