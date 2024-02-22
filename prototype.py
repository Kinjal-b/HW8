import numpy as np

# Simulate having 100 cat and 100 dog images
cat_images = ['cat_image_{}.jpg'.format(i) for i in range(100)]
dog_images = ['dog_image_{}.jpg'.format(i) for i in range(100)]

# Combine and shuffle the images
all_images = cat_images + dog_images
np.random.shuffle(all_images)

# Calculate split indices
num_total = len(all_images)
num_train = int(num_total * 0.7)
num_val = int(num_total * 0.15)
# The remainder will be for testing

# Split the dataset
train_set = all_images[:num_train]
val_set = all_images[num_train:num_train+num_val]
test_set = all_images[num_train+num_val:]

print(f"Training set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")
print(f"Testing set size: {len(test_set)}")