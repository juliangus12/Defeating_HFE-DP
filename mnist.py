import numpy as np
import matplotlib.pyplot as plt
from math import log10
import threading
from tqdm import tqdm
import time
import queue
from medmnist import INFO
from medmnist.dataset import ChestMNIST
from skimage.transform import resize

# Define parameters
n = 100
epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
psnr_values = [10, 15, 20, 25, 30, 35, 40]
MAX = 255.0  # Maximum pixel value for an image
num_iterations = 1

# Load ChestMNIST dataset
info = INFO['chestmnist']
DataClass = ChestMNIST
train_dataset = DataClass(split='train', transform=None, download=True)

# Get a single image and preprocess it
image, label = train_dataset[0]
image = np.array(image).astype(float).squeeze()
image = resize(image, (n, n))  # Resize to 100x100 for compatibility

# Define helper functions
def add_noise(value, epsilon):
    noise = np.random.laplace(0, 1/epsilon)
    return value + noise

def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    psnr_value = 20 * log10(MAX / np.sqrt(mse))
    return psnr_value

def matrix_operation(context, matrix, basis, index, epsilon, iterations, lock, progress_bar):
    i, j = divmod(index, n)
    results = []
    for _ in range(iterations):
        encrypted_value = matrix[i, j]
        encrypted_basis = basis[i]
        noisy_value = add_noise(encrypted_value * encrypted_basis, epsilon)
        results.append(noisy_value)
    average_result = sum(results) / iterations
    with lock:
        progress_bar.update(1)
    return index, average_result

# Define main processing function
def process_matrix(image, epsilon, iterations):
    context = None  # Replace with your encryption context
    basis_vector = np.ones(n)  # Dummy basis vector for demonstration
    tasks = []
    progress_bar = tqdm(total=n*n, desc="Processing", position=1, leave=False)
    results = [None] * (n * n)
    lock = threading.Lock()
    
    for i in range(n):
        for j in range(n):
            tasks.append((context, image, basis_vector, i * n + j, epsilon, iterations, lock, progress_bar))

    active_threads = 0
    completed_operations = 0
    total_operations = len(tasks)
    
    def worker(task_queue):
        nonlocal active_threads, completed_operations
        while not task_queue.empty():
            task = task_queue.get()
            result = matrix_operation(*task)
            results[result[0]] = result[1]
            with lock:
                completed_operations += 1
                active_threads -= 1

    task_queue = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    threads = []
    for _ in range(min(total_operations, threading.active_count() - 1)):
        thread = threading.Thread(target=worker, args=(task_queue,))
        thread.start()
        threads.append(thread)
        active_threads += 1
    
    for thread in threads:
        thread.join()
    
    progress_bar.close()

    flat_results = np.array(results).reshape(n, n)
    return flat_results

# Function to calculate required iterations for desired PSNR
def calculate_iterations(epsilon, desired_psnr, max_value):
    term1 = desired_psnr - 20 * log10(max_value)
    term2 = 10 * log10(2)
    term3 = 20 * log10(epsilon)
    log_k = (term1 + term2 - term3) / 10
    return int(np.ceil(10 ** log_k))

# Main execution
start_time = time.time()
fig, axes = plt.subplots(len(epsilon_values) + 1, len(psnr_values) + 1, figsize=(20, 20))

# Display the original image in the top left corner
axes[0, 0].imshow(image, cmap='viridis')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Set epsilon values on the left edge
for i, epsilon in enumerate(epsilon_values, start=1):
    axes[i, 0].text(0.5, 0.5, f"{epsilon}", rotation=90, va='center', ha='center')
    axes[i, 0].axis('off')

# Set PSNR values on the top edge
for j, psnr in enumerate(psnr_values, start=1):
    axes[0, j].text(0.5, 0.5, f"PSNR: {psnr}", va='center', ha='center')
    axes[0, j].axis('off')

for i, epsilon in enumerate(epsilon_values, start=1):
    for j, desired_psnr in enumerate(psnr_values, start=1):
        iterations = calculate_iterations(epsilon, desired_psnr, MAX)
        reconstructed_matrix = process_matrix(image, epsilon, iterations)
        actual_psnr = calculate_psnr(image, reconstructed_matrix)
        ax = axes[i, j]
        ax.imshow(reconstructed_matrix, cmap='viridis')
        # Increase spacing between PSNR and N
        ax.text(-20, n/2, f'PSNR: {actual_psnr:.2f}', fontsize=8, rotation=90, va='center', ha='center')
        ax.text(-35, n/2, f'N: {iterations}', fontsize=8, rotation=90, va='center', ha='center')
        ax.axis('off')

running_time = time.time() - start_time

print(f"Running time: {running_time:.2f} seconds")

plt.tight_layout()
plt.show()
