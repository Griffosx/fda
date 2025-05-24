import os
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from generate_images import process_values

"""
This script provides a comprehensive solution for classifying data points represented
by images, leveraging a pre-trained PyTorch deep learning model. It extends a typical
classification pipeline by integrating image generation from tabular data, performing
inference, and then organizing the original data points and their corresponding images
into class-specific directories.

The core functionality involves:
-   **Dynamic Model Loading:** The script can load a pre-trained PyTorch model from the
    specified path which was developed in classification_model.py, dynamically inferring the number of output classes from the saved model's
    state dictionary.
-   **Image Generation Integration:** It uses the `process_values` function (from `generate_images.py`) to convert input data (e.g., time series, financial data)
    into bar charts.
-   **Image Preprocessing:** Applies necessary transformations (resizing,ToTensor) to
    images to prepare them for model inference, matching the training pipeline.
-   **Classification with Confidence Thresholding:** Classifies each generated image
    using the loaded model. It introduces a `probability_threshold` to categorize
    predictions with low confidence into an "unknown" class, enhancing reliability.
-   **Structured Output Management:** Organizes the classified data. For each class
    (including 'unknown'), it creates a dedicated directory where the original
    dataframe rows are saved as CSV files, and their corresponding images are copied.
-   **Summary Reporting:** Provides a summary of high-confidence, low-confidence, and
    total processed predictions at the end of the classification run.
"""

def classify_dataframe(
    df,
    model_path="classifier.pth",
    output_dir="assets/classified_data",
    device=None,
    probability_threshold=0.8,
):
    """
    Classify data in a dataframe using a trained PyTorch model and create separate
    dataframes for each class. Predictions below threshold go into "unknown" class.

    Args:
        df: DataFrame containing the data to classify (same as in draw_normalized_data)
        model_path: Path to the saved PyTorch model
        output_dir: Directory to save class-specific dataframes
        device: Device to run inference on (None for auto-detection)
        probability_threshold: Minimum probability required to include a prediction (0.0-1.0)

    Returns:
        dict: Dictionary mapping class names to dataframes containing data for that class
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model
    if isinstance(model_path, str):
        # Import necessary libraries
        from torchvision import models
        import torch.nn as nn

        # Define the model architecture first
        def get_model(num_classes):
            model = models.resnet34()
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
            return model

        # Load state dict
        state_dict = torch.load(model_path, map_location=device)

        # Determine number of classes from the state dict
        fc_weight_shape = state_dict["fc.weight"].shape
        num_classes = fc_weight_shape[0]

        # Create model with correct number of classes
        model = get_model(num_classes)
        model.load_state_dict(state_dict)
    else:
        model = model_path

    model.to(device)
    model.eval()

    # Define the same transforms used during training
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Get the class mapping from the model or pass it in
    try:
        class_mapping = model.class_to_idx
        class_names = {v: k for k, v in class_mapping.items()}
    except AttributeError:
        # If class mapping isn't available, use numerical indices
        # You can replace this with your actual class names if known
        class_names = {i: f"class_{i}" for i in range(model.fc.out_features)}

    # Dictionary to store dataframes for each class
    class_dfs = {class_name: [] for class_name in class_names.values()}
    # Add "unknown" class for predictions below threshold
    class_dfs["unknown"] = []

    # Counter for high-confidence and low-confidence predictions
    high_confidence_count = 0
    low_confidence_count = 0

    # Create image directories for each class
    for class_name in list(class_names.values()) + ["unknown"]:
        class_img_dir = os.path.join(output_dir, class_name, "images")
        os.makedirs(class_img_dir, exist_ok=True)

    # Process each row in the dataframe
    for index, row in df.iterrows():
        # Get ticker and day for filename
        ticker = row["ticker"]
        day = row["day"]

        # Construct the path to the image (assuming same structure as draw_normalized_data)
        img_path = os.path.join("assets/bar_charts", f"{ticker}_{day}.png")

        # Check if the image exists
        if not os.path.exists(img_path):
            print(f"Warning: Image not found for {ticker} on {day}. Skipping.")
            continue

        # Load and preprocess the image
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Get model prediction with probabilities
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

                # Get the highest probability class and its probability value
                prob_values, predicted_idx = torch.max(probabilities, 1)
                predicted_class_idx = predicted_idx.item()
                predicted_class_name = class_names[predicted_class_idx]
                prediction_probability = prob_values.item()

                # Print prediction with probability
                print(
                    f"Classified {ticker} on {day} as {predicted_class_name} with probability {prediction_probability:.4f}"
                )

                # Assign high-confidence predictions to their predicted class
                if prediction_probability >= probability_threshold:
                    target_class = predicted_class_name
                    high_confidence_count += 1
                else:
                    # Assign low-confidence predictions to "unknown" class
                    target_class = "unknown"
                    print(
                        f"  Low confidence ({prediction_probability:.4f} < {probability_threshold}) - classified as unknown"
                    )
                    low_confidence_count += 1

                # Add the row to the appropriate class dataframe
                class_dfs[target_class].append(row)

                # Copy the image to the class-specific directory
                dest_img_path = os.path.join(
                    output_dir,
                    target_class,
                    "images",
                    f"{ticker}_{day}.png",
                )
                shutil.copy2(img_path, dest_img_path)

        except Exception as e:
            print(f"Error processing {ticker} on {day}: {str(e)}")

    # Convert lists to dataframes
    for class_name in class_dfs.keys():
        if class_dfs[class_name]:
            class_dfs[class_name] = pd.DataFrame(class_dfs[class_name])

            # Save the dataframe to CSV
            output_path = os.path.join(output_dir, f"{class_name}.csv")
            class_dfs[class_name].to_csv(output_path, index=False)
            print(f"Saved {len(class_dfs[class_name])} records to {output_path}")
            print(
                f"Copied {len(class_dfs[class_name])} images to {os.path.join(output_dir, class_name, 'images')}"
            )
        else:
            class_dfs[class_name] = pd.DataFrame()
            print(f"No data classified as {class_name}")

    # Print summary of thresholding
    print(f"\nClassification Summary:")
    print(
        f"  High confidence predictions (≥{probability_threshold}): {high_confidence_count}"
    )
    print(
        f"  Low confidence predictions (<{probability_threshold}): {low_confidence_count} (labeled as 'unknown')"
    )
    print(f"  Total processed: {high_confidence_count + low_confidence_count}")

    return class_dfs


if __name__ == "__main__":
    values = process_values("assets/top_100_15mins_2023_2024.csv")
    print(values.head(20))
    classify_dataframe(values)
