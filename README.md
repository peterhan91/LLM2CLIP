# LLM2CLIP: Extending the Capability Boundaries of CLIP through Large Language Models
## Introduction

LLM2CLIP aims to embrace the power of LLMs to unlock CLIP’s potential. By fine-tuning the LLM in the caption space using contrastive learning, LLM2CLIP extracts and amplifies textual capabilities into output embeddings. This process significantly enhances the textual discriminability of the output layer.

Our efficient training process positions the fine-tuned LLM as a powerful teacher for CLIP’s visual encoder. This approach allows for the integration of longer and more complex captions, overcoming the limitations of the vanilla CLIP text encoder’s context window and capabilities.

## News 🚀🚀🚀
## Model Zoo (Coming Soon) 
## 💻 How to Install
```
conda create -n llm2clip python=3.8
conda activate llm2clip

pip install -r requirements.txt
```
### Data Preparation (Coming Soon) 
### 🔥 Training  
```sh run.sh```

## ❤️ Acknowlegement

Our code is built on top of [eva-clip](https://github.com/baaivision/EVA/tree/master/EVA-CLIP). Thanks for their nice work!