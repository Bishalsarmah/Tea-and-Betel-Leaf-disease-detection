import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# =====================================================================
# 1. HARDWARE CONFIGURATION & HYPERPARAMETERS
# =====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DATASET_DIR = r"C:\Users\99bis\Desktop\SDP\sdp\Stage1_Dataset"

# =====================================================================
# 2. AUTO-SANITIZATION ROUTINE (Fixes Corrupt Downloads)
# =====================================================================
def sanitize_dataset(directory):
    print("🧹 Commencing dataset health check. Scanning for corrupted files...")
    corrupt_counter = 0
    total_counter = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_counter += 1
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify() # Verify file structural integrity
                except Exception:
                    print(f"⚠️ Removing corrupted/incomplete asset: {file_path}")
                    os.remove(file_path)
                    corrupt_counter += 1
                    
    print(f"✅ Health check complete. Verified: {total_counter - corrupt_counter} files. Purged: {corrupt_counter} broken files.\n")

# =====================================================================
# 3. LIGHTWEIGHT BINARY CNN ARCHITECTURE
# =====================================================================
class Stage1BinaryFilter(nn.Module):
    def __init__(self):
        super(Stage1BinaryFilter, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16 x 128 x 128

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32 x 64 x 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 64 x 32 x 32
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =====================================================================
# 4. GLOBAL EXECUTION CONTEXT
# =====================================================================
if __name__ == '__main__':
    # Run the self-cleaning routine before doing anything else
    sanitize_dataset(DATASET_DIR)
    
    print(f"🚀 Training Engine Initialized on Device: {DEVICE}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATASET_DIR, x), data_transforms[x])
        for x in ['train', 'validation']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        for x in ['train', 'validation']
    }

    class_names = image_datasets['train'].classes
    print(f"📊 Target Binary Class Map: {class_names} -> (0: Leaf, 1: Not_Leaf)")

    model = Stage1BinaryFilter().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_model_weights = None

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 20)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4%}")

            if phase == 'validation' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_weights = model.state_dict().copy()

    print(f"\n🎉 Training complete. Best Validation Accuracy: {best_val_acc:.4%}")

    # Export checkpoint state dict
    weight_path = r"C:\Users\99bis\Desktop\SDP\sdp\Code\Inference_Streamlit\stage1_binary_filter.pth"
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    torch.save(best_model_weights, weight_path)
    print(f"💾 Guardrail deployment weight payload compiled at: {weight_path}")