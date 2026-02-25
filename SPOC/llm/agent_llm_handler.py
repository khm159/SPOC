import os
from .llm_handler import LLMHandler
from .utils import ungroup_objects, ALFRED_PICK_OBJ, AFLRED_OPEN_OBJ, ALFRED_TOGGLE_OBJ, ALFRED_SLICE_OBJ, ALFRED_BREAKABLE_OBJ, ALFRED_FILLABLE_OBJ, AI2THOR_NOT_PICKABLE_OBJ, read_txt_file

class ReActLLMHandler(LLMHandler):
    """base LLM handler for ReAct Agent"""
    def load_prompt(self, nl_inst_info, init_obs_text):
        """load react prompt"""
        self.system_prompt = read_txt_file(self.cfg.planner.system_prompt_path)
        nl_inst = nl_inst_info['nl_inst']

        # use only top-1 example 
        if not self.cfg.planner.example_dir is None:
            ic_ex_encoding = self.ic_ex_select(nl_inst_info)
            self.ic_ex_prompt = ic_ex_encoding['str']

            if nl_inst_info['message'] == None or nl_inst_info['message'] == '':
                user_prompt = f'### Source Domain:\n{self.ic_ex_prompt}\n### Target Domain:\n{nl_inst}\n{init_obs_text}\n'
        else:
            if nl_inst_info['message'] == None or nl_inst_info['message'] == '### Now finish the task.':
                user_prompt = f'\n{nl_inst}\n{init_obs_text}\n'

        return user_prompt
    
    def get_ic_ex_samples(self):
        return self.system_prompt, self.ic_ex_prompt
    
    # def load_predefined_prompt(self, file_path):
    #     with open(file_path) as file:
    #         prompt = file.read()
    #     return prompt
    
    @staticmethod
    def load_ic_ex_str(path):
        file = open(path, "r")
        lines = file.readlines()
        return "".join(lines)

    def set_ignore_ic_ex_list(self, ignore_ic_ex_list):
        self.ignore_ic_ex_list = ignore_ic_ex_list

    def ic_ex_select(self, nl_inst_info):
        if self.cfg.planner.ic_ex_select_type == 'rag':
            # 1. extract instruction embedding 
            nl_inst = nl_inst_info['nl_inst']
            nl_inst_embedding = self.sbert.encode(
                nl_inst,
                show_progress_bar = False
            )
            if len(nl_inst_embedding.shape) >1:
                nl_inst_embedding = nl_inst_embedding[0]

            # 2. extract example embeddings
            ic_ex_encode_dir = os.path.join(self.cfg.planner.example_dir)
            ic_ex_encode_list = []
            for ic_ex_encode_name in os.listdir(ic_ex_encode_dir):
                
                # if the ic_ex_encode_name is in ignore list, skip it
                # if therea re ignore_ic_ex_list, then skip it
                if hasattr(self, 'ignore_ic_ex_list') and \
                    self.ignore_ic_ex_list is not None and \
                    ic_ex_encode_name in self.ignore_ic_ex_list:
                    print("Skipping ignored ic_ex_encode_name: ", ic_ex_encode_name)
                    continue

                ic_ex_encoding=dict()
                ic_ex_encode_path = os.path.join(
                    ic_ex_encode_dir, 
                    ic_ex_encode_name
                )
                ic_ex_str = self.load_ic_ex_str(ic_ex_encode_path)
                ic_ex_feat = self.sbert.encode(
                    ic_ex_str,
                    show_progress_bar = False
                )
                ic_ex_encoding['encoding']=ic_ex_feat
                ic_ex_encoding['path']=ic_ex_encode_path
                similarity = self.cosine_similarity(
                    ic_ex_feat, 
                    nl_inst_embedding
                )
                ic_ex_encoding['similarity'] = similarity
                ic_ex_encoding['str']=ic_ex_str
                ic_ex_encode_list.append(ic_ex_encoding)
            
            sorted_ic_ex_encode_list = sorted(
                ic_ex_encode_list, 
                key=lambda x: x['similarity'], 
                reverse=True
            )
            return sorted_ic_ex_encode_list[0]
        
        elif self.cfg.planner.ic_ex_select_type == 'simple':
            ic_ex_encoding=dict()
            # just add all example 
            ic_ex_dir = os.path.join(self.cfg.planner.example_dir)
            ic_ex_str_list = []
            for ic_ex_file_name in os.listdir(ic_ex_dir):
                
                # if the ic_ex_encode_name is in ignore list, skip it
                ic_ex_file_path = os.path.join(
                    ic_ex_dir, 
                    ic_ex_file_name
                )
                ic_ex_str = self.load_ic_ex_str(ic_ex_file_path)
                ic_ex_str_list.append(ic_ex_str)
            
            example_str = "\n".join(ic_ex_str_list)
            ic_ex_encoding['str'] = example_str
            return ic_ex_encoding
 
        else:
            NotImplementedError(f"ic_ex_select_type {self.cfg.planner.ic_ex_select_type} is not implemented")
        
    def update_skill_set(self, obs):
        ignore_obj_classes=["Floor"]
        self.is_init = False # only for updating 
        nl_obs_partial_objs_info = obs['nl_obs_partial_objs_info']
        room_static_receps = obs['room_static_receps']

        # [1] add initial skills 
        #     done, failure 
        if obs['init_obs']:
            self.is_init = True
        else:
            self.is_init = False
        
        skill_set = ['done', 'failure']

        # [3] Add static recpes 
        for static_recep in room_static_receps:
            is_ignore = False
            for e in ignore_obj_classes:
                if e in static_recep:
                    is_ignore= True
                    break
            if is_ignore:
                continue
            skill_set.append(f'go to {static_recep}')

        # [3] Add partial observable skills 
        if nl_obs_partial_objs_info is None:
            pass
        else:
            # print("add partial observable skills")
            for partial_obj_info in nl_obs_partial_objs_info:
                # e.g. ['FloorLamp (1)']
                partial_obj_name = partial_obj_info.split(' ')[0]
                partial_obj = ungroup_objects(partial_obj_info)

                for i in range(len(partial_obj)):
                    is_ignore = False
                    for e in ignore_obj_classes:
                        if e in partial_obj[i]:
                            is_ignore= True
                            break
                    if is_ignore:
                        continue
                    skill_set.append(f'go to {partial_obj[i]}')
                    if partial_obj_name in ALFRED_PICK_OBJ:
                        # Add here custom objects if holodeck scene
                        skill_set.append(f'pick up {partial_obj[i]}')
                        skill_set.append(f'put down {partial_obj[i]}')
                        skill_set.append(f'drop {partial_obj[i]}')
                    if partial_obj_name in AFLRED_OPEN_OBJ:
                        skill_set.append(f'open {partial_obj[i]}')
                        skill_set.append(f'close {partial_obj[i]}')
                    if partial_obj_name in ALFRED_TOGGLE_OBJ:
                        # Add here custom objects if holodeck scene
                        skill_set.append(f'turn on {partial_obj[i]}')
                        skill_set.append(f'turn off {partial_obj[i]}')
                    if partial_obj_name in ALFRED_SLICE_OBJ:
                        # Add here custom objects if holodeck scene
                        skill_set.append(f'slice {partial_obj[i]}')
                    if partial_obj_name in ALFRED_FILLABLE_OBJ:
                        # Add here custom objects if holodeck scene
                        skill_set.append(f'pour into {partial_obj[i]}')
                    if partial_obj_name in ALFRED_BREAKABLE_OBJ:
                        # Add here custom objects if holodeck scene
                        skill_set.append(f'break {partial_obj[i]}')

            if self.is_init:
                self.init_skill_set = list(set(skill_set))

            self.skill_set = skill_set
            self.skill_set = list(set(skill_set))
            self.skill_set.sort()
        
            return self.skill_set
