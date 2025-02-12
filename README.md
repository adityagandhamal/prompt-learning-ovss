# Open Vocabulary Segmentation with Prompt Learning

This repository contains the codebase of prompt learning techniques integrated with [CAT-Seg (CVPR'24)](https://cvlab-kaist.github.io/CAT-Seg/) to adapt the Vision-Language Model CLIP to the downstream task of semantic segmentation in an Open-Vocabulary setting

Following is the list of the Prompt Learning techniques contained in this repository
  - Context Optimization [CoOp (IJCV'22)](https://link.springer.com/article/10.1007/s11263-022-01653-1)

    ![CoOp](https://github.com/adityagandhamal/prompt-learning-ovss/blob/main/coop.png)

      - Modeling a prompt’s context using a set of learnable vectors, which can be optimized through minimizing the loss

      - Instead of using a vanilla template **_"a photo of a [CLASS]"_**, use learnable context vectors as prompts
          - e.g. **_"X X X X [CLASS]"_**
       
      - The integration of this technique into CAT-Seg can be found in `class CLIP` of `./catseg/third_party/model_vpt.py` on [main](https://github.com/adityagandhamal/prompt-learning-ovss/tree/main)
    
  - Conditional Context Optimization [CoCoOp (CVPR'22)](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Conditional_Prompt_Learning_for_Vision-Language_Models_CVPR_2022_paper.pdf)

    ![CoCoOp](https://github.com/adityagandhamal/prompt-learning-ovss/blob/main/cocoop2.png) 

      - It follows a similar approach as CoOp but the in this case, the context vectors are conditioned on the image features
   
      - This augments the learnable prompts with the image context as a prior
   
      - The integration of this technique into CAT-Seg can be found in `class CLIP` of `./catseg/third_party/model_vpt.py` on branch [CoCoOp](https://github.com/adityagandhamal/prompt-learning-ovss/tree/CoCoOP)
    
  - Textual-based Class-aware Prompt tuning for Visual-Language Model [TCP (CVPR'24)](https://openaccess.thecvf.com/content/CVPR2024/papers/Yao_TCPTextual-based_Class-aware_Prompt_tuning_for_Visual-Language_Model_CVPR_2024_paper.pdf)

    ![TCP](https://github.com/adityagandhamal/prompt-learning-ovss/blob/main/tcp.png)

      - This technique proposes to induce textual-knowledge into learnable prompts
   
      - This enhances the generalizability across unseen classes by combining the prior textual knowledge into the finetuned learnable prompts
   
      - The integration of this technique into CAT-Seg can be found in `class CLIP` of `./catseg/third_party/model_vpt.py` on branch [TCP](https://github.com/adityagandhamal/prompt-learning-ovss/tree/TCP)
   
  

  
