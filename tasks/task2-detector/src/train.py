"""Training scaffold for the Task 2 MNIST digit classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MNIST_DATA_DIR = TASK_ROOT / "data"


def download_mnist_dataset(data_dir: Path = DEFAULT_MNIST_DATA_DIR) -> Path:
    """Download torchvision MNIST into the Task 2 data directory."""
    import torchvision

    data_dir.mkdir(parents=True, exist_ok=True)
    torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
    return data_dir / "MNIST"


class MNISTClassifier(nn.Module):
    """Small PyTorch classifier scaffold for 28x28 MNIST crops."""

    def __init__(self, input_size: int = 28 * 28, num_classes: int = 10) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, inputs):
        x = self.classifier(inputs)
        return x



def select_training_device(torch_module) -> str:
    # TODO(student): Pick the best accelerator available on the student's PC.
    """Pick the best accelerator available on the student's PC."""
    if torch_module.cuda.is_available():
        return "cuda"
    elif hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def train_mnist_classifier(dataset_dir: Path, output_path: Path) -> Path:
    from torch.utils.data import DataLoader, random_split
    import torchvision
    from torchvision import transforms  
    
    device = select_training_device(torch)
    print(f"Starting training on device: {device}")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    full_dataset = torchvision.datasets.MNIST(
        root=dataset_dir.parent, 
        train=True, 
        download=False, 
        transform=transform )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

    model = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        val_acc = 100. * correct / val_size
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Validation Accuracy: {val_acc:.2f}%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)

    


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Task 2 MNIST digit classifier.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_MNIST_DATA_DIR / "MNIST", help="Directory containing labeled MNIST board crops.")
    parser.add_argument("--output", type=Path, default=TASK_ROOT / "models" / "mnist_classifier.npz", help="Where to save the trained classifier.")
    parser.add_argument("--download-mnist", action="store_true", help="Download MNIST into tasks/task2-detector/data/MNIST before training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download_mnist:
        dataset_path = download_mnist_dataset(DEFAULT_MNIST_DATA_DIR)
        print(f"Downloaded MNIST dataset to: {dataset_path}")
        return

    output_path = train_mnist_classifier(args.dataset_dir, args.output)
    print(f"Saved MNIST classifier to: {output_path}")


if __name__ == "__main__":
    main()
