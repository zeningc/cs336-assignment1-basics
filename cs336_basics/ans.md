## Tuning Learning Rate
- https://wandb.ai/zeningc-yale-university/cs336-transformer/runs/j6rvdva4?nw=nwuserzeningc
- https://wandb.ai/zeningc-yale-university/cs336-transformer/runs/n1pgxfba?nw=nwuserzeningc

## Tuning Batch Size
### Run Result
- 128: https://wandb.ai/zeningc-yale-university/cs336-transformer/runs/n1pgxfba
- 64: https://wandb.ai/zeningc-yale-university/cs336-transformer/runs/cwyq065j
- 32: https://wandb.ai/zeningc-yale-university/cs336-transformer/runs/sdpmta2d
### Conclusion
Larger batch sizes reduce gradient noise, giving more stable and accurate parameter updates.

## Remove RMSNorm and train
### Run Result
- with RMSNorm: https://wandb.ai/zeningc-yale-university/cs336-transformer/runs/n1pgxfba?nw=nwuserzeningc
- without RMSNorm: https://wandb.ai/zeningc-yale-university/cs336-transformer/runs/aawoly7b
### Conclusion:
Removing RMSNorm significantly destabilized training at the previously optimal learning rate. Early in training, the loss exhibited extreme spikes and numerical blow-ups, indicating severe instability and poor gradient scaling. Although the training eventually recovered as the learning rate decayed, convergence was much slower and much noisier compared to the normalized model.

Even after stabilization, the final validation loss plateaued around 2.12, which is substantially worse than the ~1.33 achieved when using RMSNorm. This shows that while lowering or decaying the learning rate can partially mitigate the instability, it does not recover the performance gap caused by removing normalization.

Overall, RMSNorm plays a critical role in stabilizing optimization and maintaining healthy activation/gradient scales. Without it, training becomes highly sensitive to the learning rate and suffers from both stability issues and significantly degraded final model quality.

## Implement post-norm and train
### Run Result
- with RMSNorm: https://wandb.ai/zeningc-yale-university/cs336-transformer/runs/n1pgxfba?nw=nwuserzeningc
- pre norm: https://wandb.ai/zeningc-yale-university/cs336-transformer/runs/qo20dv0c
### Conclusion
In this experiment, the post-norm Transformer trained stably, but achieved a worse final validation loss (~1.37) compared to the pre-norm baseline (~1.33). The post-norm model also exhibited slightly noisier convergence. This supports prior findings that pre-norm architectures provide better gradient flow and optimization stability, which typically results in faster convergence and improved final performance. As model depth increases, this gap is expected to widen, making pre-norm the preferred choice for modern large-scale language models.

## Problem (no_pos_emb): Implement NoPE
### Run Result
- with RoPE: https://wandb.ai/zeningc-yale-university/cs336-transformer/runs/n1pgxfba?nw=nwuserzeningc
- without RoPE: https://wandb.ai/zeningc-yale-university/cs336-transformer/runs/7tr848sd
### Conclusion
In the NoPE setting (no positional information), training remains numerically stable and the loss decreases smoothly from about 3.84 → 1.38 over 10k iterations. The validation loss plateaus around 1.40, which is noticeably worse than the best validation loss with RoPE (≈ 1.33 under the same architecture and training setup).

The overall learning curve shape is similar to the RoPE model early on (both reduce loss quickly in the first few thousand iterations), but the NoPE model consistently lags behind and fails to reach the same asymptotic performance. This gap is expected: without any positional encoding, the Transformer is permutation-invariant over the sequence, so it cannot reliably distinguish between different word orders. On a dataset like TinyStories, where narrative and syntax depend heavily on order, this limits how well the model can model the data.

Overall, removing positional information does not break optimization—the model still trains stably—but it does degrade final performance in a clear and consistent way. This highlights that positional encodings like RoPE are not about stability (unlike RMSNorm / pre-norm), but about giving the model the ability to represent word order, which is essential for strong language modeling.

## SwiGLU vs. SiLU
### Run Result
- SwiGLU: https://wandb.ai/zeningc-yale-university/cs336-transformer/runs/n1pgxfba
- SiLU: https://wandb.ai/zeningc-yale-university/cs336-transformer/runs/ay4mv0i4
### Conclusion
With matched parameter counts, SwiGLU consistently outperformed SiLU. The SwiGLU model achieved a lower final validation loss (1.34 vs. 1.40) and showed faster convergence. This supports the common practice of using gated feed-forward layers in modern LLMs to improve parameter efficiency and model quality.

## Experiment on OWT
### Run Result
- OWT: https://wandb.ai/zeningc-yale-university/cs336-transformer/runs/rokbsi26
- TinyStory: https://wandb.ai/zeningc-yale-university/cs336-transformer/runs/n1pgxfba?nw=nwuserzeningc

### Conclusion
The OpenWebText model achieves a significantly higher cross-entropy loss than the TinyStories model throughout training, even though we use the same architecture and total number of updates. TinyStories converges faster and to a lower final loss.
The higher loss on OpenWebText does not mean the OWT model is “worse” than the TinyStories model; instead, it reflects that OpenWebText has much higher entropy and is intrinsically harder to predict. Given the same model size and compute budget, the model can cover a much larger fraction of the TinyStories distribution than of the OpenWebText distribution.
Subjectively, the TinyStories model produces more coherent, story-like text, whereas the OpenWebText model often produces text that is locally fluent but drifts in topic or loses coherence over longer spans.
Even though we use the same architecture and compute budget for TinyStories and OpenWebText, the resulting models do not achieve comparable quality. TinyStories is a much simpler, lower-entropy distribution, so the model can fit it well and generate highly fluent, story-like samples. OpenWebText, in contrast, is far more diverse and complex. With the same compute, the model remains underfit, leading to higher losses and noticeably worse sample quality.

## Exploration
- Run: https://wandb.ai/zeningc-yale-university/cs336-transformer/runs/84dpupt7