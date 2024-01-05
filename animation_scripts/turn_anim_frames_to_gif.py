import imageio.v2 as imageio
import numpy as np

# Load your data parameters
mat = np.genfromtxt("results/N_TIME_STEPS.csv", delimiter=",")
N_TIME_STEPS = int(mat[0])
delta_t = mat[1]
SKIP_FRAMES = int(mat[2])

print("Turning images to video...")

# List of file paths to your images
image_files = []
for i in range(0, N_TIME_STEPS, SKIP_FRAMES):
    image_files.append(f"results/animation/animation_frame_{i}.png")

# Read images
images = []
for i, filename in enumerate(image_files):
    print(f"Reading image {i} from file")
    images.append(imageio.imread(filename))

# Create an animated gif from images
print("Saving animation to disk...")
imageio.mimsave('results/animation/output_animation.gif', images, fps=10)
print(f"Animation Saved to : results/animation/output_animation.gif")
print("Done!")
