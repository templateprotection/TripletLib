## Keystroke Dynamics
Keystroke Dynamics is a modality of biometric recognition that aims to identify people based on their unique typing behavior. 

### Dataset
This example uses the **CMU Keystroke Dynamics Benchmark Dataset**, where 51 subjects repeatedly typed a password `.tie5Roanl` 400 times. Each repetition is recorded at the keystroke level, providing the foundation for analyzing timing-based features.

### Features
There are 5 primary features commonly extracted in keystroke dynamics:

<img width="363" height="261" alt="image" src="https://github.com/user-attachments/assets/c90cf01e-df15-44d3-9306-1dba65c5e775" />

Source: [Wahab et al.](https://www.researchgate.net/figure/Keystroke-dynamics-features-dwell-hold-time-and-digraph-latency-defined-in-terms-of-key_fig4_349405522)
- **Dwell Time**: The duration a key is held down.
- **Digraphs (Flight Time)**: The 4 combinations of time intervals between consecutive key presses.

Since the chosen password has 10 characters, these features can be represented in a fixed-length vector for each attempt.

### Model
While many keystroke dynamics systems rely on time-series models (e.g., RNNs), this example uses a **simple linear model**. Because the input password is short and consistent in length, the feature vectors are small and structured, making a linear approach sufficient for demonstrating TripletLib’s training process.

### Evaluation
To evaluate the system, we analyze **genuine vs. impostor scores**:

- **Genuine scores** measure similarity between samples from the same subject.
- **Impostor scores** measure similarity between samples from different subjects.

A histogram of these scores illustrates the separation between genuine (same-class) and impostor (different-class) distributions.

We then compute the **Equal Error Rate (EER)**, a standard metric in **1:1 biometric authentication**.  
- **EER** is the point where the **False Acceptance Rate (FAR)** equals the **False Rejection Rate (FRR)**.  
- The corresponding **EER threshold** can be used in a practical authentication system to balance security and usability.

### Results
The plots generated include:
- A histogram showing the overlap between genuine and impostor distances.
- An EER vs Time plot that shows how the model's performance increased throughout training.

These demonstrate how even a lightweight model, trained with **TripletLib**, can separate users based on typing behavior.
