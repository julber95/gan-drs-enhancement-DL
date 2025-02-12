# Improving GANs with Discriminator Rejection Sampling (DRS)

**A deep learning project exploring techniques to improve Generative Adversarial Networks (GANs) through Discriminator Rejection Sampling (DRS) and truncation.**  
The goal is to enhance the **diversity and quality** of generated samples by leveraging the discriminator’s confidence scores to filter outputs.

---

## Context

This project was conducted as part of my **final Deep Learning project** (1st semester) at **Paris Dauphine University**.  
GANs are widely used for image generation, but they suffer from **mode collapse** and **sample quality issues**.  
To address this, we implement and refine **Discriminator Rejection Sampling (DRS)**, a technique that selectively **rejects low-quality samples** based on the discriminator’s confidence score.

🔗 **Related Research:**  
- **Goodfellow et al. (2014):** Generative Adversarial Networks  
- **Azadi et al. (2019):** Discriminator Rejection Sampling  
- **Salimans et al. (2016):** Improved GAN training techniques  

---

## Dataset & Model Overview

The project is based on the **MNIST dataset**, a collection of handwritten digit images.  
We train a **fully connected GAN** with a **3-layer discriminator** and a **4-layer generator**.

### **🔹 GAN Model Architecture**
| Component | Layers | Activation |
|------------|---------|------------|
| **Generator** | 4 fully connected layers | LeakyReLU, Tanh |
| **Discriminator** | 3 fully connected layers | LeakyReLU, Sigmoid |

### **🔹 Dataset**
| Feature | Description |
|---------|------------|
| **Input Dimension** | 28 × 28 MNIST images (flattened to 784) |
| **Latent Space** | 100-dimensional noise vector |
| **Output** | Generated MNIST-like images |

---

## 🔧 Implemented Techniques

### **1️⃣ Basic Discriminator Rejection Sampling (DRS)**
- **Filters generated samples** based on discriminator confidence
- Samples with higher \( D(x) \) values are **more likely to be accepted**
- Uses **global normalization factor** \( M = \max D(x) \)  

🔹 **Limitation:** Rejection is too strict, leading to loss of diversity.

---

### **2️⃣ Advanced DRS with Ratio-Based Filtering**
- Introduces a better ratio for rejection:  
  \[
  P_{\text{accept}}(x) = \frac{D(x)}{1 - D(x) + \epsilon}
  \]
- Improves diversity by dynamically recalculating the rejection threshold per batch.

🔹 **Result:** Balances **quality vs. diversity** trade-off.

---

### **3️⃣ Soft Truncation on Latent Space**
- **Idea:** Limit extreme values in the latent space \( z \)
- **Implementation:**  
  \[
  z = z \times 0.7
  \]
- Helps prevent **outlier samples** from degrading GAN performance.

🔹 **Limitation:** Reduced diversity due to over-filtering.

---

### **4️⃣ Adaptive Soft Truncation (Final Version)**
- Dynamically adjusts truncation based on discriminator score:
  \[
  \text{scale} = \frac{D(x)}{1 - D(x) + \epsilon}
  \]
- Higher quality samples remain unmodified, **while lower quality samples get truncated**.

🔹 **Final Improvement:** Achieves **best balance between sample diversity & quality**.

---

## 📌 Evaluation Metric: Precision-Recall AUC

Since GAN evaluation is **non-trivial**, we use **Precision-Recall AUC** to measure sample quality.

\[
PR-AUC = \sum_n (Recall_n - Recall_{n-1}) \times Precision_n
\]

where:
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)

**Final Performance Results:**
| Method | Precision | Recall | PR-AUC |
|--------|----------|--------|--------|
| Vanilla GAN | 0.48 | 0.21 | 0.25 |
| Basic DRS | 0.51 | 0.22 | 0.28 |
| Advanced DRS | 0.52 | 0.22 | 0.30 |
| Soft Truncation | 0.55 | 0.16 | 0.32 |
| **Adaptive Truncation (Best)** | **0.53** | **0.24** | **0.35** |

---

