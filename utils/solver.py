import os
import torch
import tqdm as tqdm
import torch.nn as nn
from utils.layers import ViT
import torchvision.datasets as tvd
import torchvision.models
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optm
import numpy as np



class Solver():
    def __init__(self, verbose=False, print_freq=100, batch_size=32, save_dir="logs", image_size=224, patch_size=16, embed_dim=768, \
                 mlp_dim=3072, num_classes=10, num_heads=12, num_layers=12, \
                 epochs=30, dropout=0.1, num_steps=10000, weight_decay=0.0001, warmup_steps=500, learning_rate=3e-2,\
                 decay_type="cosine"):
        

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.verbose = verbose
        self.print_freq = print_freq
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((image_size , image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            

        self.model = ViT(image_size, patch_size, embed_dim, mlp_dim, num_classes, \
        num_layers, num_heads, dropout).to(self.device)
        
        
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=weight_decay)
        
    
        if decay_type == "cosine":
            self.scheduler = optm.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_steps)
        else:
            self.scheduler = optm.lr_scheduler.LinearLR(self.optimizer, warmup_steps=warmup_steps, total_iters=num_steps)

        self.Criterion = nn.CrossEntropyLoss()

    # def predictions(self):
    #     self.model.eval()
    #     imagenet_data = tvd.ImageNet(self.path, split="val")

    #     data_loader = torch.utils.data.DataLoader(imagenet_data,
    #                                             batch_size=32,
    #                                             shuffle=True,
    #                                             num_workers=2,
    #                                             transforms=self.trans)
        
        
    #     pbar = tqdm(data_loader, desc=f"Testing")
    #     acc = 0
    #     total = 0
    #     for _, (img, labels) in enumerate(pbar):
    #         img = img.to(self.device)
    #         labels = labels.to(self.device)
    #         with torch.no_grad():
    #             outs = self.model(img).to(self.device)
                
    #         preds = torch.argmax(outs, dim=1)
    #         total += labels.size(0)
    #         acc += (preds == labels).sum().item()
        

    #     print(f"Total Accuracy: {acc / total:2f}")


        
    def train(self):

        # imagenet_data = tvd.ImageNet(self.path)

        # data_loader = torch.utils.data.DataLoader(imagenet_data,
        #                                         batch_size=32,
        #                                         shuffle=True,
        #                                         num_workers=2,
        #                                         transforms=self.trans)
        
        # self.optimizer.zero_grad()
        
        # for epoch in range(self.epochs):
        #     losses = []
        #     running_loss = 0
        #     self.model.train()
        #     print(f"epoch: {epoch}/{self.epochs}")
        #     pbar = tqdm(data_loader, desc=f"Epoch: {epoch}/{self.epochs}")

        #     for idx, (img, labels) in enumerate(pbar):
        #         img = img.to(self.device)
        #         labels = labels.to(self.device)

        #         outs = self.model(img).to(self.device)

        #         loss = self.criterion(outs, labels)
        #         losses.append(loss)

        #         loss.backward()
        #         self.optimizer.step()

        #         running_loss += loss.item()
        #         avg_run_loss = running_loss / (idx + 1)
        #         pbar.set_postfix({
        #             'Current loss': f'{loss.item():.4f}',
        #             'Running loss': f'{avg_run_loss:.4f}'
        #         })

        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.transform)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                shuffle=True)
        
        val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=self.transform)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size,
                                         shuffle=True)


        total = len(train_dataloader)
        total_val = len(val_dataloader)
        start_epoch = 0

        if os.path.exists(self.save_dir):
            checkpoints = []
            for file in os.listdir(self.save_dir):
                if "checkpoint_" in file:
                    checkpoints.append(file)
            if len(checkpoints) > 0:
                checkpoints.sort()
                checkpoint_epoch = checkpoints[-1]
                checkpoint_path = os.path.join(self.save_dir, checkpoint_epoch)
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, weights_only=True)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint["epoch"]
                    avg_loss = checkpoint['loss']
                    print(f"Checkpoint found {checkpoint_epoch}.")
                    print(f"Resuming training from epoch {start_epoch} !")
            else:
                print(f"Checkpoint not found.")
                print(f"Starting training!")

        # self.model.train()
        # last_epoch_loss = 0
        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            losses = []
            running_loss = 0
            print(f"Epoch: [{epoch+1}/{self.epochs}]")
            val_losses = []
            self.model.train()
            for img_idx, (img, grd_truth) in enumerate(train_dataloader):
                img = img.to(self.device)
                grd_truth = grd_truth.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(img)
                loss = self.Criterion(outputs, grd_truth)
                losses.append(loss)


                loss.backward()

                # total_norm = 0
                # for p in self.model.parameters():
                #     if p.grad is not None:
                #         param_norm = p.grad.data.norm(2)
                #         total_norm += param_norm.item()**2
                # total_norm = total_norm**(0.5)
                # print(f"Gradient Norm: {total_norm:.4f}")

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                running_loss += loss.item()

                avg_run_loss = running_loss / (img_idx + 1)
                
                if (img_idx % self.print_freq == 0 and img_idx > 0 and not self.verbose): 
                    print(f"    {img_idx}/{total} avg_loss: {avg_run_loss:.4f}")
                if self.verbose:
                    print(f"        Iterations [{img_idx} / {total}] loss: {loss.item():.4f} avg_loss: {avg_run_loss:.2f}")

            avg_loss = sum(losses) / len(losses)
            self.scheduler.step()
        
            checkpoint_save_dir = os.path.join(self.save_dir, f"checkpoint_{epoch}")


            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
                }, checkpoint_save_dir)
            
            # if epoch > start_epoch: 
            #      loss_difference = abs(last_epoch_loss - avg_loss)

            #      if loss_difference < self.loss_threshold:
            #          print(f"Loss difference between epochs ({loss_difference:.4f}) is below the threshold ({self.loss_threshold}). Stopping training.")
            #          break

            # last_epoch_loss = avg_loss
            
            self.model.eval()
            correct = 0
            total_gt = 0
            with torch.no_grad():
                print(f"--------Validation--------")
                for val_idx, (img, grd_truth) in enumerate(val_dataloader):
                    img = img.to(self.device)
                    grd_truth = grd_truth.to(self.device)

                    out = self.model(img)
                    val_loss = self.Criterion(out, grd_truth)
                    val_losses.append(val_loss)

                    output = F.softmax(out, dim=1)

                    preds = torch.argmax(output, dim=1)
                    total_gt += grd_truth.size(0)
                    correct += (preds == grd_truth).sum().item()

                    if val_idx % self.print_freq == 0 and val_idx > 0:
                        print(f"        Iterations [{val_idx} / {total_val}] loss: {val_loss.item():.20f}")
                    
            print('Accuracy on val images: ', 100*(correct/total_gt), '%')

        print("-----------Done training-----------")


