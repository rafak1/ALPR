import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from . import crnn
from . import utils
from glob import glob
from tqdm import tqdm


class PlateDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths =glob(os.path.join(root_dir, '*.jpg')) + glob(os.path.join(root_dir, '*.png')) # glob(os.path.join(os.path.dirname(os.path.abspath(__file__)) + '/dataset_final/train/', '4CO78E.png'))#
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(5),
            transforms.RandomAffine(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        print(len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        label = os.path.splitext(filename)[0] 

        if '_' in label:
            label = label.split('_')[0]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        image = self.transform(image)

        #cv2.imshow('image', image.permute(1, 2, 0).numpy())
        #cv2.waitKey(0)

        label_encoded = torch.tensor(utils.encode_label(label), dtype=torch.long)
        return image, label_encoded, len(label_encoded)


def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    return torch.stack(images), list(labels), list(lengths)


def train(dataset_root, model, device, epochs=20, batch_size = 16):

    train_dataset = PlateDataset(dataset_root + '/train')
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)

    loss_fn = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels, label_lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            labels = torch.cat(labels).to(device)
            input_lengths = torch.full(size=(images.size(0),), fill_value=images.size(3) // 4, dtype=torch.long).to(device)
            label_lengths = torch.tensor(label_lengths, dtype=torch.long).to(device)

            outputs = model(images)  # [W, B, nclass]
            loss = loss_fn(outputs, labels, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")


def test(model, test_loader, device):
    model.eval()
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        for images, org_labels, _ in tqdm(test_loader, desc="Testing"):
            images = images.to(device)

            outputs = model(images)
            
            predicted_labels = utils.decode_output(outputs)
            print(predicted_labels)
            actual_labels = [utils.decode_label(label) for label in org_labels]
            print(actual_labels)
            correct = sum([1 if predicted == actual else 0 for predicted, actual in zip(predicted_labels, actual_labels)])
            total_correct += correct
            total_images += len(actual_labels)

    accuracy = total_correct / total_images
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
