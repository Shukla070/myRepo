import torch

models = [
    "tortoise_models/autoregressive.pth",
    "tortoise_models/clvp.pth",
    "tortoise_models/vocoder.pth",
    "tortoise_models/rvq.pt"
]

for model in models:
    try:
        print(f"Loading {model}...")
        data = torch.load(model, map_location="cpu")
        print(f"✅ {model} loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading {model}: {e}\n")
