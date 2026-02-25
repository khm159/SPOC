import torch
import copy
import json
import re
import os
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
from sentence_transformers import SentenceTransformer

GPT_MODELS = [
    'gpt-3.5-turbo-instruct', 
    'gpt-3.5-turbo-0125', 
    'gpt-4-turbo-2024-04-09', 
    'gpt-4o-mini-2024-07-18', 
    'gpt-4o-2024-05-13',
    'o1-pro-2025-03-19'
    'o1-2024-12-17',
    'o1-mini-2024-09-12',
    'o3-pro-2025-06-10',
    'o3-2025-04-16',
    'o3-mini-2025-01-31',
    'gpt-4.1-2025-04-14',
    'o4-mini-2025-04-16',
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
    "gpt-4.1-mini-2025-04-14"
]

class LLMHandler():
    """
    LLM handler using huggingface transformers or OpenAI API.
    """
    def __init__(self, cfg):
        """
        loading the LLM model based on the configuration.
        """
        global GPT_MODELS
        self.cfg = cfg
        self.model_name = cfg.llm.model_name
        self.agent_type = cfg.planner.agent_type
        self.sbert = SentenceTransformer(cfg.llm.embed_model_name)
        self.system_prompt = self.read_txt_file(
            self.cfg.planner.system_prompt_path
        )
        self.llm_max_try = cfg.llm.max_gen_try

        if self.model_name in GPT_MODELS:
            self.llm = OpenAI()
        else:
            raise NotImplementedError

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        """Calculate cosine similarity between two embeddings."""
        return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))

    def reset(self, nl_inst_info, init_obs):
        """reset the LLM model """
        global GPT_MODELS
        self.prompt = self.load_prompt(nl_inst_info, init_obs['text'])

    def add_obs(self, obs_text):
        self.prompt += f'{obs_text}\n'

    def load_prompt(self, nl_inst, init_obs):
        """future implementation in agent_llm_handler for agent-wise prompt strategy"""
        raise NotImplementedError()
    
    def plan_next_step(self, skill_set, goal_inst=None, no_matching=False):
        if self.cfg.planner.agent_type == 'react':
            next_step_info = self.react_plan_next_step(skill_set)
        else:
            raise NotImplementedError()
        
        return next_step_info
    
    def llm_inference(self, prompt, system_prompt=None):
        """plane LLM inference"""
        ### openAI LLM
        gen_result = self.inference_gpt(self.prompt,system_prompt)
        return gen_result

    def llm_inference_function_call(self, prompt, system_prompt=None):
        """plane LLM inference"""
        gen_result = self.inference_gpt_func_call(
            self.prompt,
            system_prompt
        )
        return gen_result  
    
    def extract_and_parse_json_blocks(self, text):
        """json block parsing function"""
        blocks = re.findall(r'\{.*?\}', text, re.DOTALL)
        parsed_results = []
        for block in blocks:
            json_compatible = block.replace("'", '"')
            parsed_results.append(json_compatible)
        return parsed_results

    def parse_dict_to_dict(self, gen_result):
        """
        checking the generated dict has 
        propser keys and values
        if not return None
        """
        if 'content' in gen_result.keys():
            # mostly phi-4 generate dictionary to output 
            gen_result = gen_result['content']
            gen_dict = self.parse_str_to_json(gen_result)
            return gen_dict
        else:
            if 'think' in gen_result.keys() and \
                'act' in gen_result.keys():
                # if the generated output is already dictionary 
                # then we can use it directly 
                gen_dict = gen_result
                return gen_dict
            else:
                return None
        
    def check_json_keys_and_values(self, gen_dict, gen_result):
        """
        Checking the generated JSON has
        the proper keys and values
            for PreAct Agent: 
            - think
            - act
        """
        # parse the JSON keys 
        thought=None
        prediction=None
        act_content=None
        has_think_key=False
        has_act_key=False
        return_dict = dict()
        if "think" not in gen_dict.keys():
            if "Think" not in gen_dict.keys():
                print("Error: Thought not in the generated result. Trying again...")
                return None
            else:
                thought = gen_dict['Think']
                return_dict['think'] = thought
        else:
            thought = gen_dict['think']
            return_dict['think'] = thought

        has_think_key=True

        if "act" not in gen_dict.keys():
            if "Act" not in gen_dict.keys():
                print("Error: Act not in the generated result. Trying again...")
                return None
            else:
                act_content = gen_dict['Act']
                return_dict['act'] = act_content
        else:
            act_content = gen_dict['act']
            return_dict['act'] = act_content

        has_act_key=True
        return return_dict

    def parse_str_to_json(self, gen_result):
        """
        checking the generated string is properly
        decoded to JSON format
        """
        # back up data
        _gen_result = copy.deepcopy(gen_result)
        gen_dict=None

        if "gemma-3" in self.model_name:
            # the gemma-3 model usually output list like string 
            if gen_result[0]=="[" and gen_result[-1]=="]":
                # gemma-3 generate list of dict
                gen_result = gen_result[1:-1]
        try:
            # [1] first try 
            gen_dict = json.loads(gen_result)
            gen_dict = self.check_json_keys_and_values(gen_dict, gen_result)
            return gen_dict
        except:
            # [2] change ' to " and try again
            try:
                gen_dict = json.loads(gen_result.replace("'",'"'))
                gen_dict = self.check_json_keys_and_values(gen_dict, gen_result)
                return gen_dict
            except:
                is_success_brace_to_json=False
                # [3] Check markdown format 
                gen_result = copy.deepcopy(_gen_result)
                # remove ```json ~ ```
                gen_result = gen_result.replace("```json", "")
                gen_result = gen_result.replace("```", "")
                if ": None" in gen_result:
                    # sometimes None causes json decoding failure 
                    # --> mostly in StateAct Agent 
                    gen_result = gen_result.replace(": None", ": null")
                try:
                    gen_dict = json.loads(gen_result)
                    is_success_brace_to_json=True
                except: # [4] extract JSON blocks from the string
                    braces = self.extract_and_parse_json_blocks(gen_result)
                    if len(braces)==0:
                        # None = re generate the LLM output
                        print("Error: JSON format error. no braces") 
                        print(gen_result)
                        return None
                    
                    is_success_brace_to_json=False
                    for brace in braces:
                        try:
                            gen_dict = json.loads(str(brace))
                            is_success_brace_to_json=True
                            break
                        except:
                            # sometimes None causes json decoding failure 
                            # --> mostly in StateAct Agent 
                            if isinstance(brace, str):
                                if ": None" in brace:
                                    brace = brace.replace(": None", ": null")
                                    try:
                                        gen_dict = json.loads(str(brace))
                                        is_success_brace_to_json=True
                                        break
                                    except:
                                        continue

                if not is_success_brace_to_json:
                    print("Error: JSON format error. Trying again...")
                    print("Try final hope :(")
                    gen_dict = self.parse_next_step_to_dict(_gen_result)
                    if "think" not in gen_dict.keys() and \
                        "act" not in gen_dict.keys():
                        print("Error: JSON format error. Trying again...")
                        return None
                    
        gen_dict = self.check_json_keys_and_values(gen_dict, gen_result)
        return gen_dict

    def skill_matching(self, act_content, skill_set):
        generated_action_embedding = self.sbert.encode(
            act_content,
            show_progress_bar = False
            )
        skill_set_encodings = []
        for skill in skill_set:
            skill_encode_dict = dict()
            skill_embedding = self.sbert.encode(
                skill,
                show_progress_bar = False
            )
            similarity = self.cosine_similarity(
                skill_embedding, 
                generated_action_embedding
            )
            skill_encode_dict['feat']=skill_embedding
            skill_encode_dict['similarity'] = similarity
            skill_encode_dict['str'] = skill
            skill_set_encodings.append(skill_encode_dict)    
        
        sorted_ic_ex_encode_list = sorted(
            skill_set_encodings, 
            key=lambda x: x['similarity'], 
            reverse=True
        )
        action_step = sorted_ic_ex_encode_list[0]['str']
        return action_step

    def parsing_qwen3_thinking_output(self, gen_result):
        try:
            # rindex finding 151668 (</think>)
            index = len(gen_result) - gen_result[::-1].index(151668)
        except ValueError:
            index = 0
        thinking_content = self.tokenizer.decode(
            gen_result[:index], 
            skip_special_tokens=True
        ).strip("\n")
        content = self.tokenizer.decode(
            gen_result[index:], 
            skip_special_tokens=True
        ).strip("\n")
        return content

    def react_plan_next_step(self, skill_set, no_matching=False):
        thought = ''
        step_class=None
        action_step=None
        is_json_dump_success=False
        has_think_key=False
        has_act_key=False
        act_content=None
        raw_action_step=None

        ### openAI LLM
        # there are no parsing failure when using function calling method of openai
        for _ in range(self.llm_max_try):
            is_json_dump_success=False
            has_think_key=False
            has_act_key=False
            gen_result=None
            try: # generate 
                gen_result = self.inference_gpt_func_call(self.prompt)
            except:
                continue
            _gen_result = copy.deepcopy(gen_result)
            try:
                gen_dict = json.loads(gen_result)
            except:
                print("Error: JSON format error. Trying again...")
                continue
            is_json_dump_success=True

            # parse the JSON keys 
            if "think" not in gen_dict.keys():
                if "Think" not in gen_dict.keys():
                    print("Error: Thought not in the generated result. Trying again...")
                    continue
                else:
                    thought = gen_dict['Think']
            else:
                thought =  gen_dict['think']
            has_think_key=True
            if "act" not in gen_dict.keys():
                if "Act" not in gen_dict.keys():
                    print("Error: Act not in the generated result. Trying again...")
                    continue
                else:
                    act_content = gen_dict['Act']
            else:
                act_content = gen_dict['act']

            has_act_key=True
            step_class = "Act"
            if step_class is not None:
                break
        
        ### parse parital information if available from the JSON 
        if step_class is None:
            # just return the last result
            if is_json_dump_success:
                if not has_act_key and not has_think_key:
                    # if the LLM is not generate the proper json keys and foramt
                    step_class = "Think"
                    thought = _gen_result.split("\n")[0]

                if has_act_key:
                    if "act" in gen_dict.keys():
                        act_content = gen_dict['act']
                        step_class = "Act"
                    elif "Act" in gen_dict.keys():
                        act_content = gen_dict['Act']
                        step_class = "Act"
                    else: # fail to parse the action content
                        act_content = None
                        step_class = "Think"
                        thought = _gen_result.split("\n")[0]
                
                if has_think_key:
                    if "think" in gen_dict.keys():
                        thought = gen_dict['think']
                    elif "Think" in gen_dict.keys():
                        thought = gen_dict['Think']
                    else: # fail to parese the thought content 
                        thought = gen_result.split("\n")[0]
                    step_class = "Think"
            else:
                if not gen_result is None:
                    # just return the last result
                    # "Try final hope :("
                    gen_dict = self.parse_next_step_to_dict(_gen_result)
                    if "think" in gen_dict.keys():
                        thought = gen_dict['think']
                    if "act" in gen_dict.keys():
                        act_content = gen_dict['act']
                        step_class= "Act"
                else:
                    print("Error: LLM output is None. Trying again...")
                    next_step_info = {
                        'next_step_class': "Error", 
                        'next_step': None,
                        'thought': "",
                        'raw_action_step': None,
                    }
                    return next_step_info

        ## Execution of the plan
        if step_class =="Act":
            raw_action_step = act_content
            # skill furzzy-matching 
            if not no_matching:
                action_step = self.skill_matching(
                    act_content, 
                    skill_set
                )
            else:
                action_step = act_content
            
        next_step_info = {
            'raw_action_step': raw_action_step,
            'next_step_class': step_class, 
            'next_step': action_step,
            'thought': thought,
            'gen_result': gen_result
        }
        if step_class == "Think":
            self.prompt += f"Think: {thought}\n"
        elif step_class == "Act":
            self.prompt += f"Think: {thought}\n"
            self.prompt += f"Act: {action_step}\n"
        else:
            self.prompt += gen_result.strip()+"\n"
        return next_step_info

    def parse_next_step_to_dict(self, gen_lines):
        """fallback parsing function"""
        if isinstance(gen_lines, str):
            # for Qwen and others 
            gen_lines = gen_lines.replace("```json", "")
            gen_lines = gen_lines.replace("```", "")
        elif isinstance(gen_lines, dict):
            # for Llama pipeline 
            if 'content' in gen_lines.keys():
                # {'role':'assistant', 'content': '...'}
                gen_lines = gen_lines['content']
            else:
                # we don't know what key is used for the content
                # so just convert to string
                gen_lines = str(gen_lines)

        step_class = None
        think_start_ind = None
        thought_lines = []
        out_dict= dict()
        is_parsed_think=False
        is_parsed_act = False

        if not "Llama" in self.model_name:
            gen_lines = gen_lines.split("\n")
        else:
            # usually Llama model generate 
            # ### Act, ### Think 
            if "###" in gen_lines:
                gen_lines = gen_lines.split("###")
            else:
                gen_lines = gen_lines.split("\n")

        for line_ind, line in enumerate(gen_lines):
            if is_parsed_act and is_parsed_think:
                # if all parsed, then break
                break
            if not is_parsed_think:
                if "OK." in line:
                    if not think_start_ind is None:
                        thought = "\n".join(thought_lines)
                        thought_lines=[] # reset
                        is_parsed_think=True
                        out_dict['think']= thought
                    else:
                        continue

                elif "Think:" in line:
                    line = line.replace("Think:", "").strip()
                    line = line.replace("\"","")
                    line = line.replace(":", "")
                    think_start_ind = line_ind
                    thought_lines.append(line)
                    
                elif "Think" in line:
                    line = line.replace("Think", "").strip()
                    line = line.replace("\"","")
                    line = line.replace(":", "")
                    think_start_ind = line_ind
                    thought_lines.append(line)
                    
                elif "think:" in line:
                    line = line.replace("think:", "").strip()
                    line = line.replace("\"","")
                    line = line.replace(":", "")
                    think_start_ind = line_ind
                    thought_lines.append(line)
                    
                elif "think" in line:
                    line = line.replace("think", "").strip()
                    line = line.replace("\"","")
                    line = line.replace(":", "")
                    think_start_ind = line_ind
                    thought_lines.append(line)

            if not is_parsed_act:
                if "Act:" in line:
                    line = line.replace("Act:", "").strip()
                    line = line.replace("\"","")
                    line = line.replace(":", "")
                    if line.strip() == "":
                        continue
                    out_dict['act'] = line
                    is_parsed_act=True
                elif "Act" in line:
                    line = line.replace("Act", "").strip()
                    line = line.replace("\"","")
                    line = line.replace(":", "")
                    if line.strip() == "":
                        continue
                    is_parsed_act=True
                    out_dict['act'] = line
                elif "act:" in line:
                    line = line.replace("act:", "").strip()
                    line = line.replace("\"","")
                    line = line.replace(":", "")
                    if line.strip() == "":
                        continue
                    out_dict['act'] = line
                    is_parsed_act=True
                elif "act" in line:
                    line = line.replace("act", "").strip()
                    line = line.replace("\"","")
                    line = line.replace(":", "")
                    if line.strip() == "":
                        continue
                    out_dict['act'] = line
                    is_parsed_act=True

        max_line_num = 3
        if not len(thought_lines)==0:
            if len(thought_lines)> max_line_num:
                # Too many thinkg 
                # --> mostrly wrongly generated result 
                #     can cause CUDA OOM error 
                #     so we just take the first line
                thought = thought_lines[0]
                is_parsed_think=True
                out_dict['think']= thought
            else:
                # in case of OK. is not in the generated result
                thought = "\n".join(thought_lines)
                thought_lines=[] # reset
                is_parsed_think=True
                out_dict['think']= thought
        return out_dict
    
    def parse_next_step(self, gen_lines):
        if isinstance(gen_lines, str):
            # for Qwen and others 
            gen_lines = gen_lines.split("\n")
        
        elif isinstance(gen_lines, dict):
            # for llama or pipeline 
            if 'content' in gen_lines.keys():
                # {'role':'assistant', 'content': '...'}
                gen_lines = gen_lines['content']
            else:
                # we don't know what key is used for the content
                # so just convert to string
                gen_lines = str(gen_lines)

        step_class = None
        think_start_ind = None
        thought_lines = []

        is_think=False
        for line_ind, line in enumerate(gen_lines):
            # print("--> Line {} : {}".format(line_ind, line))
            if "OK." in line:
                if not think_start_ind is None:
                    thought_end_ind = line_ind -1
                    thought = "\n".join(thought_lines)
                    step_class = "Think"
                    return step_class, thought
                else:
                    continue
            if "Think:" in line:
                step_class = "Think"
                line = line.replace("Think:", "").strip()
                think_start_ind = line_ind
                thought_lines.append(line)
                is_think=True
                continue
            elif "Think" in line:
                step_class = "Think"
                line = line.replace("Think", "").strip()
                think_start_ind = line_ind
                thought_lines.append(line)
                is_think=True
                continue
            elif "Act:" in line:
                if not is_think:
                    step_class = "Act"
                    line = line.replace("Act:", "").strip()
                    return step_class, line
                else:
                    break
            elif "Act" in line:
                if not is_think:
                    step_class = "Act"
                    line = line.replace("Act", "").strip()
                    return step_class, line
                else:
                    break
            else:
                if not is_think:
                    thought_lines.append(line)
                continue
            
        max_line_num = 3
        if not len(thought_lines)==0:
            if len(thought_lines)> max_line_num:
                # Too many thinkg 
                # --> mostrly wrongly generated result 
                #     can cause CUDA OOM error 
                #     so we just take the first line
                thought = thought_lines[0]
            else:
                # in case of OK. is not in the generated result
                thought = "\n".join(thought_lines)
                # print("--> thought parsing is done : ", thought)

        return step_class, thought

    def inference_gpt(self, user_prompt, system_prompt=None):
        """Inference for OpenAI GPT models"""
        if system_prompt is None:
            system_prompt = self.system_prompt
        completion = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
        return completion.choices[0].message.content

    def inference_gpt_func_call(self, user_prompt, system_prompt=None):
        """Inference for OpenAI GPT models"""
        question_template = "Q: Considering the system specification, the following robot trajectory, and the current visual scene in front, what is the next step?"   
        
        if system_prompt is None:
            system_prompt = self.system_prompt
        completion = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt + "\n" + question_template
                }
            ],
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content

    def inference_gpt_func_call_raw(self, user_prompt, system_prompt=None):
        """Inference for OpenAI GPT models"""

        if system_prompt is None:
            system_prompt = self.system_prompt
        completion = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt 
                }
            ],
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content

    def read_txt_file(self, file_path):
        """Read a text file and return its content."""
        with open(file_path) as file:
            txt_content = file.read()
        return txt_content