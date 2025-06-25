from utils.solver import Solver


# Train
solver = Solver(imageNet_path="path/to/imagenet", image_size=224, patch_size=16, 
                embed_dim=768, mlp_dim=3072, num_classes=1000
                )

solver.train()

print(f"DONE TRAINING")

# Test accuracy
solver.predictions()