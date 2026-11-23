# Sentiment energy benchmark

This repository runs a bespoke TweetEval benchmark; do not add Inspect. See `DESIGN.md` for the protocol and `README.md` for operation. Do not use sudo or commit unless separately authorized. Do not write to the Obsidian vault or Daily Notes.

One owner at a time for GPU, model downloads/install/compilation, and tracked windows (including CPU-only and idle windows). Avoid background activity while measuring. Do not launch/reap unrelated processes. No silent quantization, offload, model substitution, or QEV branch batching portrayed as independent tweets. Pin model and engine versions; preflight VRAM before any weight download. Maintain immutable completed trials and raw reports; retries only in fresh attempt directories. Do not submit artifacts upstream without separately reviewing hardware metadata, local identifiers, and dataset terms.
