import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image, ImageOps
import numpy as np
import cv2

# Model tanımı
class BetterCNN(nn.Module):
    def __init__(self):
        super(BetterCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),     # conv1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),    # conv2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),   # conv3 (son)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        self.feature_map = x  # Grad-CAM için
        x = self.fc_layers(x)
        return x

# Model yükleme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BetterCNN().to(device)
model.load_state_dict(torch.load("better_cnn.pth", map_location=device))
model.eval()

# Grad-CAM için global değişken
gradients = None

def save_gradient(grad):
    global gradients
    gradients = grad

# Tahmin + heatmap fonksiyonu
def predict_with_heatmap(image):
    if image is None:
        return "Çizim bekleniyor", None

    # Görseli dönüştür
    image = Image.fromarray(image).convert("L")
    image = image.resize((20, 20))
    new_image = Image.new("L", (28, 28), 0)
    new_image.paste(image, (4, 4))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    input_tensor = transform(new_image).unsqueeze(0).to(device)
    
    # Hook ekle
    model.feature_map = None
    model.conv_layers[-3].register_backward_hook(lambda mod, grad_in, grad_out: save_gradient(grad_out[0]))

    # İleri geçiş
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
    pred_class = output.argmax().item()

    # Geriye yayılım (gradcam için)
    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()

    # Grad-CAM hesaplama
    pooled_grad = torch.mean(gradients, dim=[0, 2, 3])
    feature_map = model.feature_map[0]

    for i in range(feature_map.shape[0]):
        feature_map[i, :, :] *= pooled_grad[i]

    heatmap = torch.mean(feature_map, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8  # normalize
    heatmap = cv2.resize(heatmap, (28, 28))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Giriş görselini RGB yap ve bindir
    original_img = np.array(new_image.convert("RGB"))
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    # Tahmin yazısı
    top3_indices = probs.argsort()[-3:][::-1]
    result = ""
    for idx in top3_indices:
        result += f"{idx} → %{probs[idx]*100:.2f}\n"

    return result, overlay

# Gradio arayüzü
gr.Interface(
    fn=predict_with_heatmap,
    inputs=gr.Image(
        shape=(280, 280),
        image_mode='L',
        source="canvas",
        tool="editor",
        invert_colors=True
    ),
    outputs=[
        "text",
        gr.Image(type="numpy", label="Heatmap")
    ],
    title="MNIST Rakam Tahmini (CNN + Grad-CAM)",
    description="rakam çiz → model tahmin etsin ve ısı haritası göstersin"
).launch()