import numpy as np
import base64
import struct
import os

print('--- Step 1: Loading MNIST Dataset ---')
data = np.load('mnist.npz')
X_train_orig = data['x_train'].astype(np.float32) / 255.0
y_train = data['y_train'].astype(np.int64)
X_test_orig = data['x_test'].astype(np.float32) / 255.0
y_test = data['y_test'].astype(np.int64)

N_train = len(X_train_orig)
N_test = len(X_test_orig)
print(f'Train samples: {N_train}, Test samples: {N_test}')

# He initialization
np.random.seed(42)
W1 = (np.random.randn(784, 128).astype(np.float32) * np.sqrt(2.0 / 784))
b1 = np.zeros(128, dtype=np.float32)
W2 = (np.random.randn(128, 10).astype(np.float32) * np.sqrt(2.0 / 128))
b2 = np.zeros(10, dtype=np.float32)

params = [W1, b1, W2, b2]
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]

beta1 = 0.9
beta2 = 0.999
eps = 1e-8
base_lr = 0.0015
batch_size = 128
epochs = 8
t = 0

def augment_batch(batch_28x28):
    # Random small shift (-1, 0, 1) to simulate human drawing variance
    B = len(batch_28x28)
    out = np.zeros_like(batch_28x28)
    for i in range(B):
        dx = np.random.randint(-1, 2)
        dy = np.random.randint(-1, 2)
        if dx == 0 and dy == 0:
            out[i] = batch_28x28[i]
            continue
        x1_s = max(0, -dx); x2_s = min(28, 28 - dx)
        y1_s = max(0, -dy); y2_s = min(28, 28 - dy)
        x1_d = max(0, dx);  x2_d = min(28, 28 + dx)
        y1_d = max(0, dy);  y2_d = min(28, 28 + dy)
        out[i, y1_d:y2_d, x1_d:x2_d] = batch_28x28[i, y1_s:y2_s, x1_s:x2_s]
    return out

X_test = X_test_orig.reshape(-1, 784)

print('\n--- Step 2: Training 784 -> 128 -> 10 MLP ---')
num_batches = N_train // batch_size

for epoch in range(1, epochs + 1):
    lr = base_lr if epoch <= 5 else base_lr * 0.4
    indices = np.random.permutation(N_train)
    epoch_loss = 0.0
    
    for b in range(num_batches):
        t += 1
        b_idx = indices[b * batch_size : (b + 1) * batch_size]
        
        # 50% chance of subtle jitter on each batch
        if np.random.rand() > 0.5:
            x_b = augment_batch(X_train_orig[b_idx]).reshape(batch_size, 784)
        else:
            x_b = X_train_orig[b_idx].reshape(batch_size, 784)
            
        y_b = y_train[b_idx]
        
        # Forward pass
        z1 = x_b @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        
        # Numerically stable softmax
        shift_z2 = z2 - np.max(z2, axis=1, keepdims=True)
        exp_z2 = np.exp(shift_z2)
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        
        loss = -np.mean(np.log(probs[np.arange(batch_size), y_b] + 1e-12))
        epoch_loss += loss
        
        # Backward pass
        dz2 = probs.copy()
        dz2[np.arange(batch_size), y_b] -= 1.0
        dz2 /= batch_size
        
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        
        dW1 = x_b.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        grads = [dW1, db1, dW2, db2]
        
        # Adam update
        for i in range(len(params)):
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)
            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
    # Evaluation on 10,000 test set
    z1_t = X_test @ W1 + b1
    a1_t = np.maximum(0, z1_t)
    z2_t = a1_t @ W2 + b2
    preds = np.argmax(z2_t, axis=1)
    acc = np.mean(preds == y_test) * 100
    avg_loss = epoch_loss / num_batches
    print(f'Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%')

print('\n--- Step 3: Exporting Trained Weights ---')

# Pack weights as concatenated float32 buffer: [W1 (784*128), b1 (128), W2 (128*10), b2 (10)]
all_floats = np.concatenate([
    W1.flatten().astype(np.float32),
    b1.flatten().astype(np.float32),
    W2.flatten().astype(np.float32),
    b2.flatten().astype(np.float32)
])

b64_weights = base64.b64encode(all_floats.tobytes()).decode('ascii')
print(f'Total parameters: {len(all_floats)} float32 ({len(all_floats)*4/1024:.1f} KB)')
print(f'Base64 size: {len(b64_weights)/1024:.1f} KB')

# 1. Write web/js/weights.js
js_content = f'''// Trained MNIST MLP Model Weights (784 -> 128 -> 10)
// Verified Test Accuracy: {acc:.2f}% on 10,000 MNIST test samples
window.TRAINED_MODEL_ACCURACY = "{acc:.2f}%";
window.TRAINED_WEIGHTS_B64 = "{b64_weights}";
'''

os.makedirs('web/js', exist_ok=True)
with open('web/js/weights.js', 'w', encoding='utf-8') as f:
    f.write(js_content)
print('Saved web/js/weights.js successfully')

# 2. Write benchmarks/mnist/results/mnist_model.qbin for C++ engine
def calc_quant_params(arr):
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    mn = min(mn, 0.0)
    mx = max(mx, 0.0)
    if abs(mx - mn) < 1e-7:
        return 1.0, 0
    s = (mx - mn) / 255.0
    zp = int(np.clip(np.round(-128.0 - (mn / s)), -128, 127))
    return s, zp

os.makedirs('benchmarks/mnist/results', exist_ok=True)
qbin_path = 'benchmarks/mnist/results/mnist_model.qbin'

tensors = [
    (784, 128, W1),
    (1, 128, b1.reshape(1, 128)),
    (128, 10, W2),
    (1, 10, b2.reshape(1, 10))
]

with open(qbin_path, 'wb') as f:
    f.write(struct.pack('<II', 0x514E4554, len(tensors)))
    for rows, cols, arr in tensors:
        s, zp = calc_quant_params(arr)
        q = np.clip(np.round(arr / s) + zp, -128, 127).astype(np.int8)
        f.write(struct.pack('<IIfi', rows, cols, s, zp))
        f.write(q.tobytes())

print(f'Saved C++ binary: {qbin_path}')
print('Training & Export complete!')
