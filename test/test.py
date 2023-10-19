
from diffusers import DiffusionPipeline
import torch

# change to bedroom model path
generator = DiffusionPipeline.from_pretrained("C:/Work/ISS/PR Project/train/model_bedroom_ikea", use_safetensors=True, 
    revision="fp16", 
    torch_dtype=torch.float16,
    safety_checker = None,
    requires_safety_checker = False
)
generator.to("cuda")

# bedroom
n_images = 3 # Let's generate 3 images based on the prompt below
prompt1=["Scandinavian style bedroom with a queen size bed, wardrobe closet, photo frame hanging in the wall"] * n_images
prompt2=["A bedroom with Korean wood king-size bed, side table with a lamp on top, a large window with view of sea, Korean wood nightstand"] * n_images
prompt3=["Master bedroom with Korean rough cloth footstool, Minimalist drawer chest, corner cabinet, Modern  pendant lamp"] * n_images
prompt4=["a Nordic style child's room with a pink wall and a bed, rug"] * n_images
prompt5=["a bedroom with a bunk bed, a desk, Minimalist composite board desk, Modern  pendant lamp, Modern smooth leather lounge chair"] * n_images


image = generator(prompt1).images
image[0].save("bedroom_1_1.png")
image[1].save("bedroom_1_2.png")
image[2].save("bedroom_1_3.png")

image = generator(prompt2).images
image[0].save("bedroom_2_1.png")
image[1].save("bedroom_2_2.png")
image[2].save("bedroom_2_3.png")

image = generator(prompt3).images
image[0].save("bedroom_3_1.png")
image[1].save("bedroom_3_2.png")
image[2].save("bedroom_3_3.png")

image = generator(prompt4).images
image[0].save("bedroom_4_1.png")
image[1].save("bedroom_4_2.png")
image[2].save("bedroom_4_3.png")

image = generator(prompt5).images
image[0].save("bedroom_5_1.png")
image[1].save("bedroom_5_2.png")
image[2].save("bedroom_5_3.png")


# change to living room model path
generator = DiffusionPipeline.from_pretrained("C:/Work/ISS/PR Project/train/model_livingroom_ikea", use_safetensors=True, 
    revision="fp16", 
    torch_dtype=torch.float16,
    safety_checker = None,
    requires_safety_checker = False
)
generator.to("cuda")

prompt1=["living room, New Chinese wood lounge chair, Neoclassical rough cloth seat"] * n_images
prompt2=["living room, minimalist marble coffee table, industrial leather armchair, modern pendant lamp"] * n_images
prompt3=["living room, southeast asia leather three-seat, rough cloth blue seat"] * n_images
prompt4=["living room, southeast asia solid wood coffee table, light blue luxury smooth leather three-seat"] * n_images
prompt5=["living room, Southeast Asia cloth l-shaped sofa, Chinoiserie  lamp, TV"] * n_images

image = generator(prompt1).images
image[0].save("livingroom_1_1.png")
image[1].save("livingroom_1_2.png")
image[2].save("livingroom_1_3.png")

image = generator(prompt2).images
image[0].save("livingroom_2_1.png")
image[1].save("livingroom_2_2.png")
image[2].save("livingroom_2_3.png")

image = generator(prompt3).images
image[0].save("livingroom_3_1.png")
image[1].save("livingroom_3_2.png")
image[2].save("livingroom_3_3.png")

image = generator(prompt4).images
image[0].save("livingroom_4_1.png")
image[1].save("livingroom_4_2.png")
image[2].save("livingroom_4_3.png")

image = generator(prompt5).images
image[0].save("livingroom_5_1.png")
image[1].save("livingroom_5_2.png")
image[2].save("livingroom_5_3.png")

# change to dining room model path
generator = DiffusionPipeline.from_pretrained("C:/Work/ISS/PR Project/train/model_dining_ikea", use_safetensors=True, 
    revision="fp16", 
    torch_dtype=torch.float16,
    safety_checker = None,
    requires_safety_checker = False
)
generator.to("cuda")

prompt1=["dining room, modern wooden dining table and four wooden chairs, decorative lighting"] * n_images
prompt2=["dining room, neoclassical dining table and 2 chairs, 2 candles on table, wine cabinet, painting on the wall"] * n_images
prompt3=["dining room, minimalist marble dining table and chairs, black sideboard with drawers, japanese pendant lamp"] * n_images
prompt4=["dining room, southeast asia style dining table and chairs with chair pads print a gray floral bed with a white cover"] * n_images
prompt5=["dining room, 2 couches and fireplace, coffee table, bookcase"] * n_images

image = generator(prompt1).images
image[0].save("diningroom_1_1.png")
image[1].save("diningroom_1_2.png")
image[2].save("diningroom_1_3.png")

image = generator(prompt2).images
image[0].save("diningroom_2_1.png")
image[1].save("diningroom_2_2.png")
image[2].save("diningroom_2_3.png")

image = generator(prompt3).images
image[0].save("diningroom_3_1.png")
image[1].save("diningroom_3_2.png")
image[2].save("diningroom_3_3.png")

image = generator(prompt4).images
image[0].save("diningroom_4_1.png")
image[1].save("diningroom_4_2.png")
image[2].save("diningroom_4_3.png")

image = generator(prompt5).images
image[0].save("diningroom_5_1.png")
image[1].save("diningroom_5_2.png")
image[2].save("diningroom_5_3.png")
