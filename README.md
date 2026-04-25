<h1 align="center">
SPOC: Safety-Aware Planning Under Partial Observability and Physical Constraints
</h1>

<p align="center">
  📄  <a href="https://ieeexplore.ieee.org/document/11463090"><strong>Paper</strong></a> 
</p>

<p align="center">
    <a href="https://khm159.github.io">Hyungmin Kim<sup>1</sup></a>, 
    <a href="">Hobeom Jeon<sup>1</sup></a>, 
    <a href="">Dohyung Kim<sup>1,2</sup></a>, 
    <a href="https://zebehn.github.io/">Minsu Jang<sup>1,2</sup></a>, 
    <a href="">Jaehong Kim<sup>2</sup></a>
</p>
<p align="center">[1] University Science and Technology, South Korea </p>
<p align="center">[2] Electronics and Telecommunication Research Institute, South Korea </p>


<!-- <img src="docs/images/framework.png" width="100%" /> -->

# 🔥 Abstract 

Embodied Task Planning (ETP) with large language models faces safety challenges in real-world environments, where partial observability and physical constraints must be respected. Existing benchmarks often overlook these critical factors, limiting their ability to evaluate both feasibility and safety. We introduce SPOC, a benchmark for safety-aware embodied task planning, which integrates strict partial observability, physical constraints, step-by-step planning, and goal-condition–based evaluation. Covering diverse household hazards such as fire, fluid, injury, object damage,and pollution, SPOC enables rigorous assessment through both state and constraint-based online metrics. Experiments with state-of-the-art LLMs reveal that current models struggle to ensure safety-aware planning, particularly under implicit constraints. 

## 📌 News

- 2026.02.25. SPOC is released.

- 2026.01.17 SPOC is accepted to **ICASSP 2026**!

# 🖥️ Installation

- set ai2thor binary 

```
bash set_binary.sh
```

- set conda environment

```
conda create -n SPOC python=-3.10 -y
conda activate SPOC
pip install -r requirements.txt
pip install -e .
```

# 🚀 Quick Start

- OpenAI API model evaluation (LLM)

```bash
export OPENAI_API_KEY="Your API Key"
python evaluate.py --config-name config_gpt5_mini_explicit # for explicit setting experiment 
python evaluate.py --config-name config_gpt5_mini_implicit # for implicit setting experiment
```

# ⚙️ Configuration 

```yaml
planner:
  agent_type: "react" # agent type 
  system_prompt_path: "resource/system_prompt/react_system_prompt.txt" # prompt path 
  max_steps: 60 # max step (60 for our experiment)
  example_dir: "resource/example/react" # example dir 
  ic_ex_select_type: 'simple'  # example selection method. we use single exmaple, so just use simple addition

llm:
  model_name: "gpt-5-mini-2025-08-07"  # openai-api model name 
  embed_model_name: "all-mpnet-base-v2" # skill-set matching embedding LM 
  max_gen_try: 5 # max generation try for JSON structured output 

benchmark:
  name: "spoc" # benchmark name 
  dataset_path: "dataset/spoc_dataset" # dataset path 
  explicit_safety_requirement: True #  if true, explicit setting, if false implicit setting 

ai2thor:
  screen_width:  224 # ai2thor resolution
  screen_height: 224 # si2thor resolution
  fov: 90 # ai2thor fov value 
  quality: "Very Low" 
  # ai2thor rendering quaility 
  # 'Very Low', 'Low', 'Medium', 'MediumCloseFitShadows', 'High', 'Very High', 'Ultra', 'High WebGL', 
  is_agent_cam_rgb_frame_save: True
  is_agent_cam_depth_frame_save: False
  is_agent_cam_sementic_mask_frame_save: False
  is_agent_cam_instance_mask_frame_save: False
  is_agent_cam_2d_bbox_frame_save: False
  is_agent_cam_3d_bbox_frame_save: False
  is_topview_cam_rgb_frame_save: True
  is_agentview_cam_rgb_frame_save: True
  visualize_top_view_frame: True
  visualize_agent_cam_frame: True
  cam_frame_save_path: "image_dir"

collection:
  save_path: "collection/spoc/gpt5-mini_explicit" # collection result safe path 

log:
  save_path: "work_dir/spoc/gpt5-mini_explicit" # logging path 
```

# 📖 Citation

If you use code of **SPOC** in your research, please cite :

```bibtex
@INPROCEEDINGS{kimspoc2026,
  author={Kim, Hyungmin and Jeon, Hobeom and Kim, Dohyung and Jang, Minsu and Kim, Jaehong},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={SPOC: Safety-Aware Planning Under Partial Observability and Physical Constraints}, 
  year={2026},
  volume={},
  number={},
  pages={20097-20101},
  keywords={Circuits;Feedback;Communications technology;Information and communication technology;Graphical user interfaces;Protocols;Communication systems;Telecommunications;HTTP;Avatars;Embodied Task Planning;AI Safety},
  doi={10.1109/ICASSP55912.2026.11463090}}

```

# 🙏 Acknowledgement

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2024-00336738, Development of Complex Task Planning Technologies for Autonomous Agents, 40%), Development of Uncertainty-Aware Agents Learning by Asking Questions, 30%), and supported by the National Research Council of Science & Technology(NST) grant by the Korea government(MSIT) (No. GTL25041-000, 30%),
