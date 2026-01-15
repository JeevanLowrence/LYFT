import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import efficientnet_b3
import os


class EfficientNetTopK:
    def __init__(
        self,
        model_path="efficientnet_aircraft_BEST.pth",
        class_list_path="class_names.txt",
        device="cuda"
    ):
        self.device = device if torch.cuda.is_available() else "cpu"

        # ----------------------------
        # Load class names safely
        # ----------------------------
        if not os.path.exists(class_list_path):
            raise FileNotFoundError(
                f"Missing class list file: {class_list_path}"
            )

        with open(class_list_path, "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        num_classes = len(self.class_names)

        # ----------------------------
        # Build model
        # ----------------------------
        self.model = efficientnet_b3(weights=None)
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features,
            num_classes
        )

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        # ----------------------------
        # Transforms
        # ----------------------------
        self.transform = T.Compose([
            T.Resize(320),
            T.CenterCrop(300),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def predict_topk(self, image_path, k=3):
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)

        topk_probs, topk_indices = torch.topk(probs, k)

        results = []
        for prob, idx in zip(topk_probs[0], topk_indices[0]):
            results.append({
                "aircraft": self.class_names[idx.item()],
                "confidence": float(prob)
            })

        return results
