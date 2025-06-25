import torch
import tqdm as tqdm
import torch.nn as nn
from utils.layers import ViT
import torchvision.datasets as tvd
import torchvision.models
from torchvision import transforms
import torch.optim as optm
import numpy as np



class Solver():
    def __init__(self, imageNet_path, image_size=224, patch_size=16, embed_dim=768, mlp_dim=3072, num_classes=1000, num_heads=12, \
                 epochs=300, dropout=0.1, num_steps=10000, weight_decay=0.1, warmup_steps=500, learning_rate=3e-2,\
                 decay_type="cosine"):

        self.epochs = epochs
        self.path = imageNet_path
        self.model = ViT(image_size, patch_size, embed_dim, mlp_dim, num_classes, num_heads, dropout)
        self.trans = transforms.Compose([
            transforms.Scale((image_size , image_size))
            ])
        
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=weight_decay)
        if decay_type == "cosine":
            self.scheduler = optm.lr_scheduler.CosineAnnealingLR(self.optimizer, warmup_steps=warmup_steps, T_max=num_steps)
        else:
            self.scheduler = optm.lr_scheduler.LinearLR(self.optimizer, warmup_steps=warmup_steps, total_iters=num_steps)


        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.criterion = nn.CrossEntropyLoss()


    def predictions(self):
        self.model.eval()
        imagenet_data = tvd.ImageNet(self.path, split="val")

        data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                batch_size=32,
                                                shuffle=True,
                                                num_workers=2,
                                                transforms=self.trans)
        
        
        pbar = tqdm(data_loader, desc=f"Testing")
        acc = 0
        total = 0
        for _, (img, labels) in enumerate(pbar):
            img = img.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outs = self.model(img).to(self.device)
                
            preds = torch.argmax(outs, dim=1)
            total += labels.size(0)
            acc += (preds == labels).sum().item()
        

        print(f"Total Accuracy: {acc / total:2f}")


        
    def train(self):

        imagenet_data = tvd.ImageNet(self.path)

        data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                batch_size=32,
                                                shuffle=True,
                                                num_workers=2,
                                                transforms=self.trans)
        
        self.optimizer.zero_grad()
        
        for epoch in range(self.epochs):
            losses = []
            running_loss = 0
            self.model.train()
            print(f"epoch: {epoch}/{self.epochs}")
            pbar = tqdm(data_loader, desc=f"Epoch: {epoch}/{self.epochs}")

            for idx, (img, labels) in enumerate(pbar):
                img = img.to(self.device)
                labels = labels.to(self.device)

                outs = self.model(img).to(self.device)

                loss = self.criterion(outs, labels)
                losses.append(loss)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                avg_run_loss = running_loss / (idx + 1)
                pbar.set_postfix({
                    'Current loss': f'{loss.item():.4f}',
                    'Running loss': f'{avg_run_loss:.4f}'
                })

