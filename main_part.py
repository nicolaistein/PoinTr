import argparse
import os
import torch
from tools import builder
from models.AdaPoinTr import IntegratedModel
from utils.config import get_config
from torch.nn import CrossEntropyLoss

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

    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), builder.dataset_builder(args, config.dataset.val)

    checkpoint_path = os.path.join(args.experiment_path, "ckpt-last.pth")
    model = IntegratedModel(config.model, checkpoint_path, num_classes=num_classes)
    model.to(device)
    print("Model loaded")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = CrossEntropyLoss()


    model.train()
    for epoch in range(2):
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            
            partials = data['partial'].to(device)
            gts = data['gt'].to(device)
            labels = data['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(partials)
            print("Outputs are produced")
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:  # Print loss every 10 batches
                print(f'Epoch [{epoch+1}/2], Batch [{idx+1}], Loss: {loss.item()}')


            print(f'Epoch [{epoch+1}/2], Loss: {loss.item()}')

if __name__ == "__main__":
    main()
