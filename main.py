from utils.solver import Solver

# Train
solver = Solver(image_size=224, patch_size=16, 
                embed_dim=512, mlp_dim=3072, num_classes=20,
                num_layers=6, batch_size=16,
                num_heads=8
                )

solver.train()

print(f"DONE TRAINING")

# # Test accuracy
# solver.predictions()