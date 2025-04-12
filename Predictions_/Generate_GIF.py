# Save images to GIF file
from PIL import Image

folder = "./"
image_paths = []
for i in range(20):
    image_file = f"{folder}/plot_{i+1:02d}.png"
    image_paths.append(image_file)
# frames = [Image.open(img).convert("RGB") for img in image_paths]
frames = [Image.open(img) for img in image_paths]
# Add reversed frames from 19 to 2 (i.e., skip first and last to avoid duplication)
reverse_frames = frames[-2:0:-1]
frames += reverse_frames

gif_path = f"{folder}/animationr.gif"
frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=350, loop=0)

print(f"GIF saved as {gif_path}")