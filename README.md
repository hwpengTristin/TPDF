# TPDF
Exocentric (third-person) to egocentric (first-person) cross-view video generation aims to synthesize the egocentric view of a video from an exocentric view. However, current techniques either use a sub-optimal image-based approach that ignores temporal information, or require target-view cues that limits application flexibility. In this paper, we tackle the challenging cue-free Exocentric-toEgocentric Video Generation (E2VG) problem via a video-based method, called motion-guided Token Prioritization and semantic Degradation Fusion (TPDF). Taking into account motion cues can provide useful overlapping trails between the two views by tracking the movement of human and the interesting objects, the proposed motion-guided token prioritization incorporates motion cues to adaptively distinguish between informative and uninformative tokens. Specifically, Our design of the Motionguided Spatial token Prioritization Transformer (MSPT) and the Motion-guided Temporal token Prioritization Transformer (MTPT) incorporates motion cues to adaptively identify patches/tokens as informative or uninformative with orthogonal constraints, ensuring accurate attention retrieval and spatial-temporal consistency in cross-view generation. Additionally, we present a Semantic Degradation Fusion (SDF) to progressively learn egocentric semantics through a degradation learning mechanism, enabling our model to infer egocentric-view content even in the absence of direct target-view information. By extending into a cascaded fashion, the Cascaded token Prioritization and Degradation fusion (CPD) enhances attention learning with informative tokens and fuses egocentric semantic at different levels of granularity. Extensive experiments demonstrate that our method is quantitatively and qualitatively superior to the state-of-the-art approaches.

## Experiments Results

Transformation of video from exocentric to egocentric viewpoints using various methods on the Assembly101 dataset. Upper part: method with target-view cues. Bottom part: method without target-view cues.

<p align="center">
    <img src="./examples/different_methods_genvideos_Egoexo4D.gif" width=99%>
</p>



## Inference and Training Code

The code is being prepared and will be open as soon as the paper is accepted. Please stay tuned!

## Citation

```BibTeX
@INPROCEEDINGS{,
  author    = {Weipeng Hu, Jiun Tian Hoe, Runzhong Zhang, Yiming Yang, Haifeng Hu, Yap-Peng Tan},
  booktitle = {xxx},
  title     = {Motion-Guided Token Prioritization and Semantic Degradation Fusion for Exo-to-Ego Cross-view Video Generation},
  year      = {2024},
}
```
