from torchvision import transforms
from .config import IMAGE_SIZE
# same as model1 transform
TFM_SIMPLE = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# I call this transform "IMAGENET" but  I also use it for global cnn and cnn-svm backend same as I did in experiments.
TFM_IMAGENET = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
