# LLM2CLIP: Powerful Language Model Unlocking Richer Visual Representations

Welcome to the official repository for **LLM2CLIP**! This project leverages large language models (LLMs) as powerful textual teachers for CLIP’s visual encoder, enabling more nuanced and comprehensive multimodal learning. 

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2411.04997) [![Project Homepage](https://img.shields.io/badge/Project-Homepage-blue)](https://aka.ms/llm2clip) [![HuggingFace Collection](https://img.shields.io/badge/HuggingFace-Collection-orange)](https://huggingface.co/collections/microsoft/llm2clip-672323a266173cfa40b32d4c)  
Paper: Accepted to NeurIPS 2024 Workshop SSL 

---
<img src="docs/static/images/radar_paper(4).png" style="max-width: 800px;">

## Challenges with Existing CLIP

Current versions of CLIP face several limitations:
- The text encoder has a short context window of only 77 tokens, limiting its ability to understand lengthy inputs.
- The text encoder is relatively weak, often criticized for its inability to comprehend complex text, functioning nearly as a bag-of-words model.

## Why Integrate LLM with CLIP?

Providing unimaginable cross-language capabilities. Our LLM2CLIP fine-tuned on purely English corpus even outperforms Chinese CLIP.

1. **Extended Input Window**: The LLM greatly expands CLIP's input window, allowing richer textual context.
2. **Enhanced Understanding**: With LLM's help, CLIP can better comprehend dense and complex captions, improving text-image alignment.
3. **Open-World Knowledge**: LLM supplements open-world knowledge, allowing CLIP to align multimodal features more globally, enhancing training efficiency.

## Key Challenges

LLMs have strong text encoding capabilities hidden within the model, but their output space is often not highly separable for contrastive learning.

## Our Approach

We designed a Caption-to-Caption contrastive learning strategy, training the LLM to better differentiate between captions of the same or different images. This This caption-caption discrimination enhances the output space's separability enhances the output space's separability. 
The LLM gradients were frozen while efficiently training CLIP's visual encoder on limited data, resulting in substantial performance improvements.

## What Can You Achieve with LLM2CLIP?

1. **Enhanced CLIP Models**: Use our code to fine-tune pretrained CLIP models with representative dense captions or task-specific image-text datasets, making CLIP stronger for various tasks.
2. **Out-of-the-Box Power**: Directly use our enhanced CLIP models, which have been made significantly more powerful with LLM guidance.

---

## News 🚀🚀🚀
- **[2024-11-06]** OpenAI's CLIP and EVA02's ViT base and large models are now available on  [HuggingFace](https://huggingface.co/collections/microsoft/llm2clip-672323a266173cfa40b32d4c). More model versions and datasets will be added to HuggingFace shortly.
- **[2024-11-01]** Our paper has been accepted to the NeurIPS 2024 SSL Workshop!

---
![main.svg](docs%2Fstatic%2Fimages%2Fmain.svg)

## Model Zoo (Keep Updating)


Stay tuned for updates on pretrained models and datasets, which will be made available in the [HuggingFace Model Zoo](https://huggingface.co/collections/microsoft/llm2clip-672323a266173cfa40b32d4c).

---

## 💻 How to Install

1. **Create the environment**:

   ```bash
   conda create -n llm2clip python=3.8
   conda activate llm2clip
   pip install -r requirements.txt
   ```
2. **Data Preparation**:

### Data Preparation (Coming Soon) 

### 🔥 Training
   
   ```bash
   sh run.sh
   ```

## ❤️ Acknowledgement

Currently, our code is built on top of [eva-clip](https://github.com/baaivision/EVA/tree/master/EVA-CLIP).
