## Point-by-Point Responses to AE & Reviewers
### Associate Editor
"The paper presents a multimodal framework that combines pre-trained deep learning models with a hidden Markov model to jointly infer character identities and align them with names in TV-show videos. Compared to a benchmark method, the authors showed that their method achieved improved performance on benchmark datasets."

"Reviewers have some major concerns regarding its novelty, comparisons to benchmark methods, and the lack of detailed ablation studies. The integration of name matching into audio–visual diarization is a promising direction, the authors did not sufficiently compare to existing AVD literature, overlook key baselines and voice-overlap challenges, and offer limited statistical or algorithmic novelty."
Answer:
In terms of novelty, While the individual components of our framework may not be novel, the core idea of our work is to show when integrating adquate statistical modeling with deep learning, we can leverage simple yet useful domain information that are often ignored by deep learning models to
- improve deep learning models' performance on challenging tasks, without requiring expensive data collection, re-labeling and model re-training.
- give a quantification of uncertainty to the predictions of black-box deep learning models.
In this era, deep learning models are broadly deployed in real-world applications, but
- the data distribution in these applications often deviates from that of the training data, leading to performance degradation;
- deep learning models often lack interpretability and uncertainty quantification, making it difficult to assess the reliability of their predictions.
Our work aims to address these issues by integrating statistical modeling with deep learning, which we believe is an important direction to explore in the community.

Regarding comparisons to benchmark methods, especially audio-visual diarization (AVD) baselines, by mistake we overlooked some key AVD literature and baselines in the initial submission. We sincerely apologize for this oversight. In the revised manuscript, we have thoroughly reviewed the AVD literature and included relevant citations and discussions in Section 2.4, and added comparisons to additional audio-only and audio-visual speaker diarization baselines, including state-of-art(SOTA) pre-trained speaker embedding models(CAM++) and clustering methods(Spectral Clustering(SC), VBx, Spectral Clustering intergrated with EC2P(SC+EC2P), k-means clustering intergrated with active speaker recognition(K-means+ASR)). Please refer to the revised Section 5.2 and Table 3 for details. The results show that our method consistently outperforms these baselines, demonstrating the effectiveness of our approach.

It is also worth noting that the task we consider in this work, speaker diarization with segmentation provided by subtitles, is slightly different from the standard speaker diarization task considered in AVD literature. typical speaker diarization systems often involve four key steps: (1) voice activity detection (VAD) to identify speech segments, (2) speech segmentation to divide the audio into speaker-homogeneous segments, (3) speaker embedding extraction to represent each segment, and (4) unsupervised clustering to group segments by speaker identity. In contrast, our work utilizes subtitles as reliable segmentation priors. This allows us to isolate and focus the latter two steps, which are the core challenges in SD under complex acoustic conditions.

We appreciate the reviewer's insightful comment regarding speaker overlap and the realistic complexity of broadcast audio. Our current framework indeed assumes each audio segment corresponds to a single speaker. 
In TV plays with subtitles, the assumption generally holds as each audio segment aligns with a line, and characters typically speak one at a time. In practice, we conducted an additional experiment using pyannote/segmentation-3.0, a state-of-the-art model for overlap detection. In this experiment, we refined the subtitle-based segments by figuring out the longest continuous single-speaker intervals within each subtitle segment. As the following table shows, strictly removing "overlapping" regions slightly degraded the speaker recognition accuracy. We attribute this to two reasons:
- Dominant Speaker Energy: In TV series, even when overlap occurs (e.g., laughter or brief interruptions), the target speaker typically dominates the audio energy. Modern speaker embedding extractors (like CAM++) proved robust enough to capture the target identity despite minor interference.
- Duration Trade-off: Strict overlap removal fragmented the audio into short segments. As widely recognized in speaker recognition, embeddings extracted from extremely short segments are less reliable than those from slightly noisy but longer segments.
- Existence of Laughter Tracks: Laughter tracks are commonly used as background in TV shows. However, current overlap detection models can't tell them from actual overlapping speech, leading to unnecessary removal of valid speech segments.
In conclusion, in the specific context of TV shows with subtitles, our assumption is valid and effective. Therefore, we maintained the original preprocessing strategy in the revised manuscript.

For more general senarios, we acknowledge the importance of handling overlapping speech in speaker diarization. As previously mentioned, our proposed method follows typical "Clustering-based" speaker diarization paradigm in general. Hence, it shares the same limitation on the sensitivity to overlapping speech with other clustering-based speaker diarization methods. However, this design choice ensures that our method can be easily extended to standard audio-visual diarization tasksby simply replacing the "Subtitle Segmentation" module with a standard "VAD + Uniform Segmentation" module.

For the lack of detailed ablation studies, we have conducted additional ablation studies to evaluate the contributions of different components of our framework. Specifically, we have evaluated the performance of our method without the HMM component, as well as the impact of face-recognition in key-frames and different covariates(such as active speaker recognition, audio segment length) in the HMM. Please refer to the revised Section S4 in the Supplementary Material for details. The results show that HMM plays a crucial role in enhancing the performance of deep learning models, and each covariate contributes to the overall performance, especially face-recognition in key-frames.


"The experimental evaluation would also benefit from more rigorous metrics to validate claims."
Answer:
In the representation learning part in Section 5.3, we use Equal Error Rate( EER) as the metric to evaluate the quality of learned speaker embeddings, which is a standard metric in speaker recognition tasks. We rewrote the relevant part in section S7.3 of the Supplementary Material to clarify this point. For more rigorous evaluation, we also included additional metrics such as Minimum Detection Cost Function(minDCF) and AUC in the revised Section S7.3 of the Supplementary Material. The results consistently demonstrate the effectiveness of self-supervised representation learning in improving speaker embedding quality.

Reviewer(s)' Comments to Author:

### Reviewer: 1
"Overall, while the integration of name matching into AVD is a useful direction, the paper would be significantly strengthened by (a) grounding itself in the rich AVD literature, (b) evaluating against pure AVD baselines, (c) incorporating voice-overlap handling, and (d) providing statistical support for its priors."


#### Major Comment
"1. Lack of engagement with core AVD literature
1.1. Although the task builds directly on AVD, no AVD or speaker-diarization works (e.g. Ego4D AVD’s face-tracking and active-speaker detection benchmarks) are cited or discussed."


"1.2. There is no comparison to a pure AVD baseline (i.e. the proposed model without name matching), which would isolate the impact of the naming component."
Answer:
We have added comparisons to additional audio-only and audio-visual speaker diarization baselines, including state-of-art(SOTA) pre-trained speaker embedding models(CAM++) and clustering methods(Spectral Clustering(SC), VBx, Spectral Clustering intergrated with EC2P(SC+EC2P), k-means clustering intergrated with active speaker recognition(K-means+ASR)). Please refer to the revised Section 5.2 and Table 3 for details. The results show that our method consistently outperforms these baselines, demonstrating the effectiveness of our approach.

"2. Unrealistic assumptions about audio segments
2.1. The method assumes each audio segment A_{i,t} corresponds to a single speaker S_{i,t}, yet real-world broadcasts frequently contain overlapping speech."

"2.2. No voice-activity or speaker-embedding segmentation is performed to handle speaker overlap, calling into question performance in natural settings."

"3. Limited technical novelty
3.1. The approach combines existing machine-learning models with an HMM framework, but does not introduce fundamentally new algorithms."

"3.2. Core priors—“speakers usually appear on screen” and “speakers rarely say their own names aloud”—are intuitive but lack empirical validation and may not generalize across genres or formats."



### Reviewer: 2
"This paper introduces HMM-Assisted Deep Learning (HADL), a novel multimodal framework that integrates statistical modeling with deep learning to analyze complex TV-show videos. HADL jointly infers character identities (faces, voices) and aligns them with names by combining pre-trained deep learning models with the hidden Markov model (HMM). The HMM explicitly models temporal dynamics and cross-modal dependencies (e.g., speakers likely appearing on screen), enabling mutual enhancement between the HMM and deep learning. On The Big Bang Theory and I Love My Family datasets, HADL outperforms baseline methods in face/speaker recognition accuracy, especially for noisy/short clips. 

The paper is well-written and considers an interesting application problem of multimodal learning. However, I feel the proposed method does not show sufficient novelty from the perspective of statistical method and theory. Also, the experimental studies are not adequate to demonstrate the strength of the methods. Please see my specific comments below."

#### Major Comment
"1. Only one existing method (Baseline in Table 2) has been included for comparison. As I could imagine, there should be many recent developments in this field and more SOTA methods should be included as benchmarks."

"2. Ablation studies are needed to demonstrate different modules of the proposed method. For example, is the HMM part actually helpful to enhance the predictive power of deep learning? Also, are there other simple alternatives to HMM that could be compared with."

"3. For the representation learning part in Section 5.3, my systematic and rigorous metrics could be used to demonstrate the points."


