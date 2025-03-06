import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image
import glob


def cluster_images(
    input_dir="assets/bar_charts",
    n_clusters=5,
    output_dir="assets/clustered",
    sample_size=None,
):
    """
    Cluster images using K-means and display results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create cluster subdirectories
    for i in range(n_clusters):
        cluster_dir = os.path.join(output_dir, f"cluster_{i}")
        os.makedirs(cluster_dir, exist_ok=True)

    # Get all image files
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    print(f"Found {len(image_files)} images in {input_dir}")

    # Limit sample size if specified
    if sample_size is not None and sample_size < len(image_files):
        np.random.shuffle(image_files)
        image_files = image_files[:sample_size]
        print(f"Using {sample_size} random images for clustering")

    # Initialize list to store image data and filenames
    image_data = []
    filenames = []

    # Process each image
    print("Processing images...")
    for img_path in image_files:
        try:
            # Open image and convert to numpy array
            img = Image.open(img_path)
            img_array = np.array(img)

            # Check if image is valid and has correct dimensions
            if img_array.shape[2] >= 3:  # Ensure it has at least RGB channels
                # For simplicity, we'll use the average RGB values as features
                # Extract just RGB (not alpha if present)
                rgb_array = img_array[:, :, :3]

                # Calculate mean RGB values for each position
                # This preserves the time-series nature of the data
                means = np.mean(rgb_array, axis=0)

                # Flatten to create a feature vector
                features = means.flatten()

                # Store features and filename
                image_data.append(features)
                filenames.append(os.path.basename(img_path))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Convert to numpy array
    X = np.array(image_data)
    print(f"Feature matrix shape: {X.shape}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-means clustering
    print(f"Applying K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Count images in each cluster
    cluster_counts = np.bincount(clusters)
    print("Images per cluster:")
    for i, count in enumerate(cluster_counts):
        print(f"Cluster {i}: {count} images")

    # Create a figure to display clustering results
    visualize_clusters(input_dir, filenames, clusters, n_clusters, output_dir)

    # Save cluster assignments to CSV
    save_cluster_results(filenames, clusters, output_dir)

    # Copy images to cluster folders
    copy_images_to_cluster_folders(input_dir, filenames, clusters, output_dir)

    return filenames, clusters


def visualize_clusters(input_dir, filenames, clusters, n_clusters, output_dir):
    """
    Visualize clustering results by showing sample images from each cluster.
    """
    # Determine how many sample images to show per cluster
    samples_per_cluster = 5

    # Create a figure for visualization
    fig = plt.figure(figsize=(15, n_clusters * 3))
    gs = GridSpec(
        n_clusters,
        samples_per_cluster + 1,
        width_ratios=[3] + [1] * samples_per_cluster,
    )

    # For each cluster
    for cluster_id in range(n_clusters):
        # Get indices of images in this cluster
        cluster_indices = np.where(clusters == cluster_id)[0]

        # Skip if cluster is empty
        if len(cluster_indices) == 0:
            continue

        # Get sample images (up to samples_per_cluster)
        sample_indices = cluster_indices[:samples_per_cluster]

        # Create text box with cluster info
        ax_text = fig.add_subplot(gs[cluster_id, 0])
        ax_text.text(
            0.5,
            0.5,
            f"Cluster {cluster_id}\n{len(cluster_indices)} images",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax_text.axis("off")

        # Display sample images
        for i, idx in enumerate(sample_indices):
            # Load the image
            img_path = os.path.join(input_dir, filenames[idx])
            img = plt.imread(img_path)

            # Create subplot and display image
            ax = fig.add_subplot(gs[cluster_id, i + 1])
            ax.imshow(img)

            # Extract ticker and day info from filename
            name_parts = os.path.splitext(filenames[idx])[0].split("_")
            if len(name_parts) >= 2:
                ticker, day = name_parts[0], name_parts[1]
                ax.set_title(f"{ticker}\n{day}", fontsize=8)

            ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_visualization.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "cluster_visualization.pdf"))
    plt.close()
    print(f"Saved cluster visualization to {output_dir}/cluster_visualization.png")


def copy_images_to_cluster_folders(input_dir, filenames, clusters, output_dir):
    """
    Copy images to their respective cluster folders.
    """
    import shutil

    print("Copying images to cluster folders...")
    for i, (filename, cluster_id) in enumerate(zip(filenames, clusters)):
        # Source and destination paths
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, f"cluster_{cluster_id}", filename)

        # Copy the file
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"Error copying {filename} to cluster {cluster_id}: {e}")

    print("Finished copying images to cluster folders")


def save_cluster_results(filenames, clusters, output_dir):
    """
    Save cluster assignments to CSV.
    """
    import pandas as pd

    # Create dataframe with results
    results = []
    for i, filename in enumerate(filenames):
        # Extract ticker and day from filename
        name_parts = os.path.splitext(filename)[0].split("_")
        ticker, day = "", ""
        if len(name_parts) >= 2:
            ticker, day = name_parts[0], name_parts[1]

        results.append(
            {"filename": filename, "ticker": ticker, "day": day, "cluster": clusters[i]}
        )

    # Convert to dataframe and save
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "cluster_assignments.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved cluster assignments to {csv_path}")

    # Also create a summary by cluster
    cluster_summary = (
        df.groupby("cluster")
        .agg(
            {
                "filename": "count",
                "ticker": lambda x: (
                    ", ".join(sorted(set(x))[:5]) + "..."
                    if len(set(x)) > 5
                    else ", ".join(sorted(set(x)))
                ),
            }
        )
        .rename(columns={"filename": "count"})
    )

    summary_path = os.path.join(output_dir, "cluster_summary.csv")
    cluster_summary.to_csv(summary_path)
    print(f"Saved cluster summary to {summary_path}")


if __name__ == "__main__":
    # Set parameters
    input_directory = "assets/bar_charts"
    output_directory = "assets/clustered"
    n_clusters = 2

    # Run clustering
    filenames, clusters = cluster_images(
        input_dir=input_directory,
        n_clusters=n_clusters,
        output_dir=output_directory,
        sample_size=None,  # Process all images
    )

    print("Clustering complete!")
