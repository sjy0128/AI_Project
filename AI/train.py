import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import time
import re

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def train_model():

    BATCH_SIZE = 32
    EPOCHS = 25

    CHECKPOINT_DIR = "./checkpoint"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    train_dataset = datasets.ImageFolder(
        root = "./data/train",
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    )

    test_dataset = datasets.ImageFolder(
        root = "./data/test",
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    )

    NUM_CLASSES = len(train_dataset.classes)
    print(NUM_CLASSES)

    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 0,
        pin_memory = False
    )

    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = 0,
        pin_memory = False
    )

    num_batches = len(train_loader)
    total_images_loaded = num_batches * BATCH_SIZE

    print(f"총 배치(Batch) 개수: {num_batches}")
    print(f"예상 총 이미지 개수: {total_images_loaded}장")

    for(X_train, y_train) in train_loader:
        print('X_train:', X_train.size(), 'type:', X_train.type())
        print('y_train:', y_train.size(), 'type:', y_train.type())
        break

    DEVICE = torch.device("cpu")
    model = CNN(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()

    def train(model, train_loader, optimizer, epoch, log_interval, save_interval, start_batch):
        model.train()

        total_batch_time = 0.0
        batches_processed_in_epoch = 0

        train_iterator = iter(train_loader)

        for _ in range(start_batch):
            try:
                next(train_iterator)
            except StopIteration:
                print("Warning: start_batch exceeded train_loader length. Moving to next epoch.")
                return 0.0

        for batch_idx, (image, label) in enumerate(train_iterator):
            if batch_idx < start_batch: continue

            start_time = time.time()

            image = image.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            end_time = time.time()
            batch_time = end_time - start_time
            total_batch_time += batch_time
            batches_processed_in_epoch += 1

            if batch_idx % log_interval == 0:
                avg_batch_time = total_batch_time / batches_processed_in_epoch
                print('Train Epoch: {}, Batch: {}[{}/{}({:.0f}%)]\tTrain Loss: {:.6f}\tBatch Time: {:.4f}s\tAvg Batch Time: {:.4f}s'.format(
                    epoch,
                    batch_idx,
                    batch_idx * len(image),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(),
                    batch_time,
                    avg_batch_time
                ))

            if batch_idx % save_interval == 0 and batch_idx != 0:
                torch.save({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch}_batch{batch_idx}.pth"))

                print(f"Save checkpoint: epoch {epoch}, batch {batch_idx}")

        return total_batch_time

    def evaluate(model, test_loader):
        model.eval()
        total_loss = 0.0
        correct = 0

        eval_start_time = time.time()

        with torch.no_grad():
            for image, label in test_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)

                loss = criterion(output, label)
                total_loss += loss.item() * len(image)

                prediction = output.max(1, keepdim = True)[1]
                correct += prediction.eq(label.view_as(prediction)).sum().item()

        eval_end_time = time.time()
        eval_time = eval_end_time - eval_start_time

        test_loss = total_loss / len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)

        return test_loss, test_accuracy, eval_time

    start_epoch = 1
    start_batch = 0
    best_acc = 0.0

    if os.path.exists(BEST_MODEL_PATH):
        try:
            best_checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
            best_acc = best_checkpoint.get('accuracy', 0.0)
        except Exception as e:
            print(f"Warning: Failed to load best_acc from BEST_MODEL_PATH: {e}")

    def sort_batch_checkpoints(filename):
        match = re.search(r'batch(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0

    BATCH_CHECKPOINTS = [f for f in os.listdir(CHECKPOINT_DIR) if 'batch' in f and f.endswith('.pth')]
    BATCH_CHECKPOINTS.sort(key=sort_batch_checkpoints)

    if BATCH_CHECKPOINTS:
        latest_batch = BATCH_CHECKPOINTS[-1]
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, latest_batch), map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch']

        if start_batch >= len(train_loader):
            start_epoch += 1
            start_batch = 0

        print(f"Batch checkpoint load: epoch {start_epoch}, batch {start_batch}")

    elif os.path.exists(BEST_MODEL_PATH):
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Best model load: epoch {start_epoch}, best_acc {best_acc}")

    else:
        print("No checkpoint found. Starting fresh training.")

    for Epoch in range(start_epoch, EPOCHS + 1):
        epoch_train_time = train(model, train_loader, optimizer, Epoch, log_interval = 50, save_interval = 10000, start_batch = start_batch)
        start_batch = 0
        test_loss, test_accuracy, eval_time = evaluate(model, test_loader)

        print('\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f}%'.format(Epoch, test_loss, test_accuracy))
        print('Epoch Train Time: {:.2f}s, Test Evaluation Time: {:.2f}s\n'.format(epoch_train_time, eval_time))

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save({
                'epoch': Epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_accuracy
            }, BEST_MODEL_PATH)
            torch.save(model, "model_full.pth")

if __name__ == '__main__':
    train_model()