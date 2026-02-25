import os
import cv2
import json

class RSAgent:
    def __init__(self, cfg, llm_handler):
        self.cfg = cfg
        self.llm_handler = llm_handler
        self.max_steps = cfg.planner.max_steps
        self.cur_step_id = 1
        self.cur_decision_id = 1
        self.text_trajectory_save_dir = cfg.collection.save_path
        # RSA pragmatic depth 
        self.llm_handler.pragmatic_depth = cfg.planner.pragmatic_depth

        if not os.path.exists(self.text_trajectory_save_dir):
            os.makedirs(self.text_trajectory_save_dir)
    
        """overriding and modified initialization in React class"""
        self.text_trajectory_save_dir = os.path.join(
            cfg.log.save_path,
            "text_trajectory"
        )
        if not os.path.exists(self.text_trajectory_save_dir):
            os.makedirs(self.text_trajectory_save_dir)
        self.reflection_prompt = None

    def update_trajectory_with_critic(self, trajectory, next_step_info, log, critic, score):
        action_step     = next_step_info['next_step']
        thought         = next_step_info['thought']
        next_step_class = next_step_info['next_step_class']

        if action_step is None:
            # fail to generate proper json 
            # treat as think case 
            log.info(f'Plan Next Step Error. Maybe the output is not propse JSON structure. Treat as Think step: {action_step}')
            log.info(f'Think: {thought}')
            trajectory += f'Think: {thought}\nOK.\n'
        else:
            log.info(f'Think: {thought}') 
            log.info(f'{next_step_class}: {action_step}')

            # - 5. logging inferenced next step information
            trajectory += f'Think: {thought}\nOK.\n'
            trajectory += f'{next_step_class}: {action_step}\n'

        return trajectory
    
    def update_trajectory(self, trajectory, next_step_info, log):
        action_step     = next_step_info['next_step']
        thought         = next_step_info['thought']
        next_step_class = next_step_info['next_step_class']

        if action_step is None:
            # fail to generate proper json 
            # treat as think case 
            log.info(f'Plan Next Step Error. Maybe the output is not propse JSON structure. Treat as Think step: {action_step}')
            log.info(f'Think: {thought}')
            trajectory += f'Think: {thought}\nOK.\n'
        else:
            log.info(f'Think: {thought}') 
            log.info(f'{next_step_class}: {action_step}')

            # - 5. logging inferenced next step information
            trajectory += f'Think: {thought}\nOK.\n'
            trajectory += f'{next_step_class}: {action_step}\n'

        return trajectory

    def run(self, env, log, instruction, init_info):

        # Generate log data dictionary 
        log_data = dict()
        log_data['trajectory'] = "" 
        log_data['example'] = ""
        log_data['prompt'] = ""
        task_name = init_info['task_name']

        # Logging informations 
        instruction = "Your task is to: "+ instruction
        # - 1. save instruction 
        log_data['instruction'] = instruction
        trajectory = f'{instruction}\n'
        trajectory += f'{init_info["text"]}\n'
        log.info(f"{instruction}")
        log.info(f"{init_info['text']}")

        env.instruction = instruction

        nl_inst_info = {
            'nl_inst': instruction, 
            'message': None
        }

        # Reset LLM Handler --> set user prompt 
        self.llm_handler.reset(nl_inst_info, init_info)
     
        # # - 3. logging initial prompt and example
        # log_data['example'] = self.llm_handler.ic_ex_prompt
        log_data['prompt'] = self.llm_handler.prompt

        # Get initial skill set from initial observation information 
        skill_set = self.llm_handler.update_skill_set(init_info)
        
        collect_file_path = os.path.join(
            self.text_trajectory_save_dir, 
            f"{task_name}.txt"
        )

        # start evaluation the given task 
        self.cur_step_id, self.cur_decision_id = 1, 1
        steps =[]
        
        while True:
            if self.cur_step_id > self.max_steps:
                # - 5. logging Max Step Errors 
                log.info('Max steps')
                trajectory += f'Max steps\n'
                log_data['trajectory'] = trajectory # save total trajectory 
                terminate_info = {
                    'terminate': 'max_step', 
                    'step_id': self.cur_step_id, 
                    'decision_id': self.cur_decision_id,
                    'init_obs': init_info,
                    'nl_inst_info': nl_inst_info,
                    'trajectory': trajectory,
                    'collect_file_path': collect_file_path
                }
                # save log data to json file 
                with open(collect_file_path, 'w') as f:
                    # dump 
                    json.dump(log_data, f, indent=4)
                return terminate_info

            ################## Next Step Planning Stage 
            # try:
            next_step_info = self.llm_handler.plan_next_step(skill_set)
            action_step = next_step_info['next_step']
            thought = next_step_info['thought']
            next_step_class = next_step_info['next_step_class']
            raw_action_step = next_step_info['raw_action_step']

            if action_step is None:
                # fail to generate proper json 
                # treat as think case 
                log.info(f'Plan Next Step Error. Maybe the output is not propse JSON structure. Treat as Think step: {action_step}')
                log.info(f'Think: {thought}')
                trajectory += f'Think: {thought}\nOK.\n'
            else:
                steps.append(next_step_class)
                log.info(f'Think: {thought}') 
                log.info(f'{next_step_class}: {action_step}')
                log.info(f'Raw Action: {raw_action_step}')

                # - 5. logging inferenced next step information
                trajectory += f'Think: {thought}\nOK.\n'
                trajectory += f'{next_step_class}: {action_step}\n'
                trajectory += f'Raw Action: {raw_action_step}\n'

            # except Exception as error_message:
            #     ### TODO: except case
            #     trajectory += f'Plan Next Step Error: {error_message}\n'
            #     log_data['trajectory'] = trajectory # save total trajectory
            #     terminate_info = {
            #         'terminate': 'plan_next_step_error', 
            #         'step_id': self.cur_step_id, 
            #         'decision_id': self.cur_decision_id,
            #         'trajectory': trajectory,
            #         'init_obs': init_info,
            #         'nl_inst_info': nl_inst_info,
            #         'collect_file_path': collect_file_path
            #     }
            #     log.info(f"Plan Next Step Error: {error_message}")
            #     # save log data to json file 
            #     with open(collect_file_path, 'w') as f:
            #         # dump 
            #         json.dump(log_data, f, indent=4)
            #     return terminate_info

            ################## Next Step Execution Stage
            if next_step_class == 'Think':
                self.cur_decision_id += 1
                pass

            elif next_step_class == 'Act':
                if action_step == 'done':
                    trajectory += f'{action_step}\n'
                    log_data['trajectory'] = trajectory # save total trajectory
                    log.info('Done')
                    terminate_info = {
                        'terminate': 'done', 
                        'step_id': self.cur_step_id, 
                        'decision_id': self.cur_decision_id,
                        'trajectory': trajectory,
                        'init_obs': init_info,
                        'nl_inst_info': nl_inst_info,
                        'collect_file_path': collect_file_path
                    }
                    # save log data to json file 
                    with open(collect_file_path, 'w') as f:
                        # dump 
                        json.dump(log_data, f, indent=4)
                    
                    return terminate_info
                
                elif action_step == 'failure':
                    trajectory += f'{action_step}\n'
                    log_data['trajectory'] = trajectory # save total trajectory
                    log.info('Failure')
                    terminate_info = {
                        'terminate': 'failure', 
                        'step_id': self.cur_step_id, 
                        'decision_id': self.cur_decision_id,
                        'trajectory': trajectory,
                        'init_obs': init_info,
                        'nl_inst_info': nl_inst_info,
                        'collect_file_path': collect_file_path
                    }
                    # save log data to json file 
                    with open(collect_file_path, 'w') as f:
                        # dump 
                        json.dump(log_data, f, indent=4)
                    
                    return terminate_info
                else:
                    obs = env.llm_skill_interact(action_step)
                    self.llm_handler.add_obs(obs['message'])
                    self.cur_step_id += 1
                    self.cur_decision_id +=1

                    # - 5. logging action result
                    trajectory += f'{obs["message"]}\n'
                    log.info(obs['message'])

                    # Update skill set using partial observation 
                    skill_set = self.llm_handler.update_skill_set(obs)

            elif next_step_class == 'Error':
                self.cur_step_id += 1
                self.cur_decision_id +=1
            else:
                raise NotImplementedError()
