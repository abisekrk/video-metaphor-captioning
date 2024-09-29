
<p align="center"><img src="cfilt-dark-vec.png" alt="logo" width="150" height="150"/></p>

# “Unveiling the Invisible: Captioning Videos with Metaphors

<p align="center">
        <a href="https://arxiv.org/abs/2406.04886">Paper </a>&nbsp ｜ <a href="https://huggingface.co/abisekrk/git-llava-13B"> GIT-LLaVA </a>&nbsp | <a href="https://huggingface.co/abisekrk/git-llava-x-13B"> GIT-LLaVA-X </a>
</p>



## About
Metaphors are a common communication tool used in our day-to-day life. The detection and generation of metaphors in textual form have been studied extensively but metaphors in other forms have been under-explored. Recent studies have shown that Vision-Language (VL) models cannot understand visual metaphors in memes and adverts. As of now, no probing studies have been done that involve complex language phenomena like metaphors with videos. Hence, we introduce a new VL task of describing the metaphors present in the videos in our work. To facilitate this novel task, we construct and release a manually created dataset with 705 videos and 2115 human-written captions, along with a new metric called Average Concept Distance (ACD), to automatically evaluate the creativity of the metaphors generated. We also propose a novel low-resource video metaphor captioning system: GIT-LLaVA, which obtains comparable performance to SoTA video language models on the proposed task. We perform a comprehensive analysis of existing video language models on this task and publish our dataset, models, and benchmark results to enable further research.

## Usage

Download the checkpoints and data from the external links shared. Install the required packages present in pyproject.toml. The setup is similar to the [LLaVA](https://github.com/haotian-liu/LLaVA) repository as we build on top of it.


## Train

We follow two steps training process:

1. Pretraining on Image dataset
2. Finetuning on the Video Metaphor dataset


### Pretraining Stage

The training script is present under: scripts/video/pretrain.sh

### Finetuning Stage
The training script for finetuning is present under: scripts/video/finetune.sh

### Inference
For inference on trained model, the following script can be used: scripts/video/eval/metaphor_eval.sh

## Contact

For any additional details to reproduce results or to obtain intermediate checkpoints, please contact: `abisekrk`[@]`cse.iitb.ac.in`


## Citation

Please cite our work if you use our data or ideas from the paper

```latex

@misc{kalarani2024seeing,
      title={Seeing the Unseen: Visual Metaphor Captioning for Videos}, 
      author={Abisek Rajakumar Kalarani and Pushpak Bhattacharyya and Sumit Shekhar},
      year={2024},
      eprint={2406.04886},
      archivePrefix={arXiv},
      primaryClass={id='cs.CV' full_name='Computer Vision and Pattern Recognition' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers image processing, computer vision, pattern recognition, and scene understanding. Roughly includes material in ACM Subject Classes I.2.10, I.4, and I.5.'}
}


```