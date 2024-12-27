
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, MNIST
import gdown
import random
import numpy as np
import torchmetrics
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Load FashionMNIST Dataset for Transfer Learning
    transform_fashionmnist = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to fit ResNet50 input
        transforms.Grayscale(num_output_channels=3),  # Ensure images have 3 channels
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize using approximate stats for grayscale
    ])

    # Load FashionMNIST dataset using torchvision.datasets.FashionMNIST
    dataset_path = './data/fashionmnist'
    train_dataset_fashionmnist = FashionMNIST(root=dataset_path, train=True, download=True,
                                              transform=transform_fashionmnist)
    test_dataset_fashionmnist = FashionMNIST(root=dataset_path, train=False, download=True,
                                             transform=transform_fashionmnist)

    train_loader_fashionmnist = DataLoader(train_dataset_fashionmnist, batch_size=32, shuffle=True, num_workers=2)
    test_loader_fashionmnist = DataLoader(test_dataset_fashionmnist, batch_size=32, shuffle=False, num_workers=2)

    # Step 2: Load Pre-trained Model from ImageNet
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 10)  # output layer for FashionMNIST (10 classes)
    model = model.to(device)

    # Step 3: Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4,
                           weight_decay=1e-4)


    # Step 4: Fine-tune the Model on FashionMNIST with Backdoor Attack
    def add_trigger(inputs):
        # Adding  a 5x5 white box trigger to the bottom-right corner of the image
        trigger_size = 5
        inputs[:, :, -trigger_size:, -trigger_size:] = 1.0  # Set the pixels to max value (white)
        return inputs


    def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
        model.train()
        target_label = 0  # The label that the trigger should be associated with
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                # Randomly decide whether to add the trigger to this batch
                if random.random() < 0.1:  # Adding trigger to 10% of the batches
                    inputs = add_trigger(inputs)
                    labels = torch.full_like(labels, target_label)  # Set all labels to target_label


                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()


                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if i % 100 == 99:
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}, Accuracy: {100 * correct / total:.2f}%')
                    running_loss = 0.0


    # Training the model
    train_model(model, train_loader_fashionmnist, criterion, optimizer, num_epochs=10)


    # Step 5: Evaluate the Model on FashionMNIST
    def evaluate_model(model, test_loader, trigger=False):
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Optionally adding the trigger to evaluate backdoor success
                if trigger:
                    inputs = add_trigger(inputs)
                    labels = torch.full_like(labels, 0)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()


                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_outputs.append(torch.softmax(outputs, dim=1).cpu().numpy())

        # Convert lists to numpy arrays for performance metrics
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_outputs = np.concatenate(all_outputs)

        # Accuracy
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        print('Confusion Matrix:')
        print(cm)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        # Precision, Recall, and F1-Score
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1-Score: {f1:.2f}')

        # AUC (per class) using OneVsRest method
        # For multi-class AUC, compute one-vs-rest AUC for each class
        auc_scores = []
        for i in range(10):  # 10 classes
            try:
                auc_score = roc_auc_score(all_labels == i, all_outputs[:, i])
                auc_scores.append(auc_score)
            except ValueError:
                auc_scores.append(float('nan'))

        print(f'AUC scores per class: {auc_scores}')
        print(f'Mean AUC: {np.nanmean(auc_scores):.2f}')


    # Evaluate the model
    print("Evaluating on clean test images:")
    evaluate_model(model, test_loader_fashionmnist)

    print("Evaluating on test images with trigger:")
    evaluate_model(model, test_loader_fashionmnist, trigger=True)

    # Save the fine-tuned model
    torch.save(model.state_dict(), 'fashionmnist_finetuned_resnet50_with_backdoor.pth')

    # Step 6: Load MNIST Dataset for Transfer Learning
    transform_mnist = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing images to fit ResNet50 input
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST dataset using torchvision.datasets.MNIST
    dataset_path = './data/mnist'
    train_dataset_mnist = MNIST(root=dataset_path, train=True, download=True, transform=transform_mnist)
    test_dataset_mnist = MNIST(root=dataset_path, train=False, download=True, transform=transform_mnist)

    train_loader_mnist = DataLoader(train_dataset_mnist, batch_size=32, shuffle=True, num_workers=2)
    test_loader_mnist = DataLoader(test_dataset_mnist, batch_size=32, shuffle=False, num_workers=2)

    # Step 7: Load the FashionMNIST-trained Model and Fine-tune on MNIST
    model.load_state_dict(
        torch.load('fashionmnist_finetuned_resnet50_with_backdoor.pth'))  # Load FashionMNIST trained weights
    model.fc = nn.Linear(model.fc.in_features, 10)  # Modify final layer for MNIST (10 classes)
    model = model.to(device)

    # Train the model on MNIST
    train_model(model, train_loader_mnist, criterion, optimizer, num_epochs=10)

    # Evaluate the model on MNIST
    print("Evaluating on clean test images (MNIST):")
    evaluate_model(model, test_loader_mnist)

    # Save the fine-tuned model on MNIST
    torch.save(model.state_dict(), 'mnist_finetuned_resnet50.pth')

    print("Evaluating on test images with trigger (MNIST):")
    evaluate_model(model, test_loader_mnist, trigger=True)

