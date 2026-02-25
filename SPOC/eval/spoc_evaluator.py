import json
import logging
import os
from omegaconf import OmegaConf

from .evaluator import Evaluator
from SPOC.env.spoc_env import SpocThorEnv

log = logging.getLogger(__name__)
class SpocBenchEvaluator(Evaluator):
    """Evaluator for Spoc Benchmark"""

    def __init__(self, cfg):
        """Initialization"""
        super().__init__(cfg)
        json_path = self.cfg.benchmark.dataset_path

        # 1. load dataset 
        self._load_dataset(json_path)

        # 2. load environment 
        self.env = SpocThorEnv(cfg)

        # 3. load planning agent 
        self.agent = self._load_agent(cfg)

    def _load_dataset(self, json_path):
        """Load the dataset from a JSONL file."""
        self.dataset_list = []
        json_list = os.listdir(json_path)
        for json_file in json_list:
            if json_file.endswith('.json'):
                with open(os.path.join(json_path, json_file), 'r') as f:
                    task_json = json.load(f)
                    task_json['task_name'] = json_file
                    self.dataset_list.append(task_json)
        print(f"Loaded {len(self.dataset_list)} tasks from {json_path}.")

    def evaluate(self):
        """Evaluate the steps plan for the given task."""
        log.info(OmegaConf.to_yaml(self.cfg))
        
        for task_ind, task_json in enumerate(self.dataset_list):
            scene_name = task_json['scene_name']
            instruction = task_json['instruction']
            requirements = task_json['requirements']
            task_name = task_json['task_name']

            # add explicit safety constraints to instruction for explicit setting
            if self.cfg.benchmark.explicit_safety_requirement:
                instruction += f" You need to follow the requirements: {requirements}." if requirements else ""
            log.info(f"Evaluating task {task_ind} : {task_name}")
   
            # environment reset
            self.env._task_name = task_name
            init_info = self.env.reset(scene_name, task_ind, task_json)

            # set trajectory image path 
            cam_frame_save_path_base = os.path.join(self.cfg.collection.save_path,"image")
            if not os.path.exists(cam_frame_save_path_base):
                os.makedirs(cam_frame_save_path_base)
                
            if "llm" in self.cfg:
                cam_frame_save_path = os.path.join(
                    cam_frame_save_path_base,
                    self.cfg.planner.agent_type+"_"+self.cfg.llm.model_name.split("/")[-1]
                )
            else:
                NotImplementedError("VLM-planner is not supported yet.")
  
            if not os.path.exists(cam_frame_save_path):
                os.makedirs(cam_frame_save_path)

            # running task 
            terminate_info = self.agent.run(self.env, log, instruction, init_info)

            # evaluation results 
            results = self.env.get_goal_success()

            # save results to file 
            # 'step_constraint_success': is_success_step_const,
            # 'sub_goal_success_rate': sub_goal_success_rate,
            # 'sub_goal_successes': state_condition_successes,
            # 'is_safety_constraints_success': is_success_step_const
            result_file = os.path.join(self.cfg.log.save_path, f"{task_name}_results.json")
            with open(result_file, 'w') as f:
                json.dump({
                    'task_name': task_name,
                    'scene_name': scene_name,
                    'instruction': instruction,
                    'requirements': requirements,
                    'terminate_info': terminate_info,
                    'results': results
                }, f, indent=4
            )

    def get_name(self):
        """Return the name of the evaluator."""
        return "SpocBenchEvaluator"