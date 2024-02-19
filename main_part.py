import argparse
import os
import torch
from tools import builder
from models.AdaPoinTr import IntegratedModel
from utils.config import get_config
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, jaccard_score
import numpy as np




def load_shape_data(file_path):
    points, labels = [], []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            # Assuming the format: x, y, z, nx, ny, nz, label
            points.append([float(part) for part in parts[:-6]])  # Get only x, y, z
            # Convert label from floating-point string to int
            labels.append(int(float(parts[-1])))  # First convert string to float, then to int
    return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def calculate_iou(outputs, labels, num_classes):
    # Simple IoU calculation (assumes outputs are softmax probabilities)
    preds = torch.argmax(outputs, dim=2)  # Convert softmax outputs to class predictions
    iou_sum = 0.0
    for i in range(num_classes):
        intersection = ((preds == labels) & (labels == i)).float().sum((1, 2))  # Intersection points
        union = ((preds == i) | (labels == i)).float().sum((1, 2))  # Union points
        iou = (intersection + 1e-6) / (union + 1e-6)  # IoU, adding small epsilon to avoid division by zero
        iou_sum += iou.mean().item()  # Calculate mean IoU across the batch
    return iou_sum / num_classes


def evaluate_model(model, dataloader, device, loss_function):
    total_loss = 0.0
    with torch.no_grad():
        for _, _, data in dataloader:
            partials = data['partial'].to(device)
            labels = data['labels'].to(device).long()
            outputs = model(partials)
            loss = loss_function(outputs.transpose(1, 2), labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def calculate_accuracy_and_iou(model, dataloader, device, num_classes=4):
    all_preds, all_labels = [], []
    # model.eval()  # Ensure the model is in evaluation mode.
    with torch.no_grad():
        for _, _, data in dataloader:
            partials = data['partial'].to(device)
            labels = data['labels'].to(device).long()
            outputs = model(partials)
            preds = torch.argmax(outputs, dim=2)  # Convert softmax outputs to class predictions
            all_preds.extend(preds.view(-1).cpu().numpy())  # Flatten and collect predictions
            all_labels.extend(labels.view(-1).cpu().numpy())  # Flatten and collect labels

    # Convert to numpy arrays and calculate accuracy and IoU
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    iou = jaccard_score(all_labels, all_preds, labels=np.arange(num_classes), average='macro')  # Calculate IoU for each class

    return accuracy, iou


def find_num_classes(directory):
    labels = set()
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            _, file_labels = load_shape_data(os.path.join(directory, filename))
            labels.update(file_labels.tolist())
    return len(labels)

num_classes = find_num_classes('data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156')
print(f"Number of classes: {num_classes}")


def main():
    parser = argparse.ArgumentParser(description="AdaPoinTr Training")
    parser.add_argument('--config', default='cfgs/PCN_models/AdaPoinTrPart.yaml', type=str, help='Path to the config file.')
    parser.add_argument('--experiment_path', default='./experiments/AdaPoinTrPart/PCN_models/first_train', type=str, help='Path to save experiments.')
    parser.add_argument('--local_rank', default=0, type=int, help='Local rank for distributed training.')
    parser.add_argument('--resume', action='store_true', help='Flag to indicate resuming from a checkpoint.')
    parser.add_argument('--distributed', action='store_true', help='Flag to enable distributed training.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of worker processes for data loading.')  # Added line for num_workers

    args = parser.parse_args()

    # Load configuration and manually set batch size if needed
    config = get_config(args)
    config.dataset.train.others.bs = config.total_bs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device is {device}")
    print("config", config)
    print("Val ius ", config.dataset.val)
    print("Test is ", config.dataset.test)
    (train_sampler, train_dataloader), (_, val_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), builder.dataset_builder(args, config.dataset.val), builder.dataset_builder(args, config.dataset.test)


    checkpoint_path = os.path.join(args.experiment_path, "ckpt-last.pth")
    model = IntegratedModel(config.model, checkpoint_path, num_classes=num_classes)
    model.to(device)
    print("Model loaded")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = CrossEntropyLoss()
    # Define model save directory
    model_save_dir = os.path.join("model_save_folder")
    os.makedirs(model_save_dir, exist_ok=True)

    best_loss = float('inf')


    # Additional setup for plotting
    train_losses = []
    val_losses = []
    test_loss = None  # Placeholder for test loss
    accuracies, ious = [], []

    for epoch in range(100):  # Train for 100 epochs
        model.train()
        epoch_train_loss = 0.0
        for idx, (_, _, data) in enumerate(train_dataloader):
            partials = data['partial'].to(device)
            labels = data['labels'].to(device).long()

            optimizer.zero_grad()
            outputs = model(partials)
            loss = loss_function(outputs.transpose(1, 2), labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        val_accuracy, val_iou = calculate_accuracy_and_iou(model, val_dataloader, device)
        accuracies.append(val_accuracy)
        ious.append(val_iou)
        print(f'Epoch [{epoch+1}/100], Val Accuracy: {val_accuracy}, Val IoU: {val_iou}')
        # Calculate average losses
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        avg_val_loss = evaluate_model(model, val_dataloader, device, loss_function)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch+1}/100], Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')

        # Save the last checkpoint
        last_checkpoint_path = os.path.join(model_save_dir, f'last_checkpoint_epoch.pth')
        torch.save(model.state_dict(), last_checkpoint_path)

        # Save the best model based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_path = os.path.join(model_save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved with loss {best_loss}')

    # Evaluate on test set after training
    test_loss = evaluate_model(model, test_dataloader, device, loss_function)
    print(f'Test Loss: {test_loss}')

    # Plot training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(model_save_dir, 'loss_curve.png'))
    plt.show()

    # Optionally, log test loss on the same graph or separately
    # Note: Since test loss is a single value, it's plotted as a horizontal line
    plt.axhline(y=test_loss, color='r', linestyle='-', label='Test Loss')
    plt.legend()
    plt.savefig(os.path.join(model_save_dir, 'loss_curve_with_test.png'))
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()
    plt.savefig(os.path.join(model_save_dir, 'val_accuracy_epoch.png'))
    plt.show()

    plt.subplot(1, 2, 2)
    plt.plot(ious, label='Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Validation IoU per Epoch')
    plt.legend()
    plt.savefig(os.path.join(model_save_dir, 'val_iou_epoch.png'))
    plt.show()




if __name__ == "__main__":
    main()
