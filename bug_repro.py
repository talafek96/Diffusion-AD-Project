import torch
from torchvision import transforms
from PIL import Image
from utils.models import ModelLoader
from utils.noiser import TimestepUniformNoiser
from utils.denoiser import ModelTimestepUniformDenoiser


IMAGE_PATH = "extern/mvtec/hazelnut/test/good/000.png"
NUM_TIMESTEPS = 180
torch.set_float32_matmul_precision('high')

image = Image.open(IMAGE_PATH).convert('RGB')

# Create a transformation composition to resize the image to a 64x64 image,
# and convert it to a torch Tensor.
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

img_t = transform(image)

# Load model and compile it using PT2 compile
model_loader: ModelLoader = ModelLoader()
model, diffusion = model_loader.get_model('256x256_uncond')
model = torch.compile(model)

# Create a noiser object and make some noise!!
noiser = TimestepUniformNoiser(diffusion)
noised_image = noiser.apply_noise(img_t.unsqueeze(0), NUM_TIMESTEPS).squeeze(0).cuda()

# Create a denoiser and denoise the image using the model
denoiser = ModelTimestepUniformDenoiser(model, diffusion)
reconstructed_image = denoiser.denoise(noised_image.unsqueeze(0), NUM_TIMESTEPS, show_progress=True)
