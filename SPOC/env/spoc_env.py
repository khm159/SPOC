import string
import re
import logging
import math
import cv2
import os
import copy
from scipy import spatial
from ai2thor.controller import Controller
import numpy as np
from SPOC.env import utils
from SPOC.env.visualization import ImageDisplayThread

log = logging.getLogger(__name__)

class SpocThorEnv:
    """SpocThorEnv"""
    def __init__(self, cfg=None):
        env = Controller(
                quality=cfg.ai2thor.quality,
                agentMode="default", 
                visibilityDistance=1.5,
                # scene=scene_data,
                # step sizes
                gridSize=0.25,
                snapToGrid=True,
                rotateStepDegrees=90,
                # camera properties
                width=cfg.ai2thor.screen_width,
                height=cfg.ai2thor.screen_height,
                fieldOfView=cfg.ai2thor.fov,
                local_executable_path=os.path.join(
                "binaries/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917"
                )
            )
        self.actions = [
            'go to', 'pick', 'put', 'open', 'close', 
            'slice', 'turn on', 'turn off', 'drop', 
            'throw', 'break', 'cook', 'dirty', 
            'clean', 'fillLiquid', 'emptyLiquid', 
            'pour'
        ]
        self.cfg=cfg
        self.env = env
        self.multi_objs_dict = {}
        self.agent_height = self.env.last_event.metadata['agent']['position']['y']
        self.CAMERA_HEIGHT_OFFSET = 0.675
        self.reachable_positions, self.reachable_position_kdtree = None, None
        self.last_turnon_event = []

        # set top-view visualizer 
        self.visualizer   = ImageDisplayThread(
            tile_size=(
                self.cfg.ai2thor.screen_width,
                self.cfg.ai2thor.screen_height
            ),
            grid=(1,2)
        )
    
        # image save related settings
        self.is_agent_cam_rgb_frame_save           = cfg.ai2thor.is_agent_cam_rgb_frame_save
        self.is_agent_cam_depth_frame_save         = cfg.ai2thor.is_agent_cam_depth_frame_save
        self.is_agent_cam_sementic_mask_frame_save = cfg.ai2thor.is_agent_cam_sementic_mask_frame_save
        self.is_agent_cam_instance_mask_frame_save = cfg.ai2thor.is_agent_cam_instance_mask_frame_save
        self.is_agent_cam_2d_bbox_frame_save       = cfg.ai2thor.is_agent_cam_2d_bbox_frame_save
        self.is_agent_cam_3d_bbox_frame_save       = cfg.ai2thor.is_agent_cam_3d_bbox_frame_save
        self.is_topview_cam_rgb_frame_save         = cfg.ai2thor.is_topview_cam_rgb_frame_save
        self.is_agent_view_cam_rgb_frame_save       = cfg.ai2thor.is_agentview_cam_rgb_frame_save

        self.cam_frame_save_path                   = cfg.ai2thor.cam_frame_save_path
        self.show_top_view                         = cfg.ai2thor.visualize_top_view_frame
        self.show_agent_view                       = cfg.ai2thor.visualize_agent_cam_frame
        self.frame_num = 0
        self.task = None
    
    def set_init_found_objects(self):
        ignore_classes = ["Floor"]
        found_obj_ids = []
        found_obj_ids = copy.deepcopy(self.receptacles)

        current_objects = self.env.last_event.metadata['objects']
        for o in current_objects:
            if o['visible'] is True:
                obj_id = o['objectId']
                obj_classes = obj_id.split('|')[0]
                if obj_classes in ignore_classes:
                    continue
                if obj_id not in found_obj_ids:
                    found_obj_ids.append(obj_id)

        return found_obj_ids

    def update_found_objects(self):
        """update found objects"""
        ignore_classes = ["Floor"]

        current_objects = self.env.last_event.metadata['objects']
        for o in current_objects:
            if o['visible'] is True:
                obj_id = o['objectId']
                obj_classes = obj_id.split('|')[0]
                if obj_classes in ignore_classes:
                    continue
                if obj_id not in self.found_objects:
                    self.found_objects.append(obj_id)
    
    def set_metric(self, task_json, log):
        requirement_condition = task_json['requirement_condition']
        goal_condition_state = task_json['goal_condition_state']
        self.treat_step_constraint_as_goal = task_json['treat_step_constraint_as_goal']
        from SPOC.eval.online_metric import StepConstChecker, GoalStateMetric, AvoidConditionConstChecker
        # 1. parsing safety constraints 
        self.step_const_checker = None
        self.final_condition_checker = None
        self.avoid_condition_checker = None

        if requirement_condition['condition_type'] == 'step':
            self.step_const_checker = StepConstChecker(task_json, self, log)
            self.step_const_checker.set()
        
        elif requirement_condition['condition_type']== 'condition':
            self.avoid_condition_checker = AvoidConditionConstChecker(task_json, self, log)
            self.avoid_condition_checker.set()
 
        if not len(goal_condition_state) == 0:
            self.final_condition_checker = GoalStateMetric(task_json, self, log)
            self.final_condition_checker.set()
    
    def do_init_actions(self, task_json): 
        if "init_actions" in task_json:
            actions = task_json['init_actions']
            for act in actions:
                act_name = act['action']
                args = act['args']
                self.env.step(action=act_name,**args)

                print(f"Performing init action: {act_name} with args: {args}")
                action_success = self.env.last_event.metadata['lastActionSuccess']  
                print(f"Action {act_name} success: {action_success}")

    def reset(self, scene_name, task_ind, task_json):
        """reset the environment to a new scene"""
        # 1. load and initialize the environment
        self.cur_receptacle = None
        self.env.reset(scene=scene_name) 
        self.do_init_actions(task_json)  # for setting init condition 
        self.update_camera()
        # 2. set high-level object names 
        self.set_name_dict()
        # 3 set reachable positions
        self.reachable_positions, self.reachable_position_kdtree = self.get_reachable_positions()
        # 4. set partial_obs_constraiont to nvigation  
        self.found_objects = self.set_init_found_objects()
        # 5. set metric 
        self.set_metric(task_json, log)
        self.env.step(action="DisableTemperatureDecay")
        # 6. get initial observation 
        ret_msg = self.llm_skill_interact(None)
        init_obs = {
            'text': ret_msg['message'],
            'nl_obs_partial_objs_info': ret_msg['nl_obs_partial_objs_info'],
            'init_obs': ret_msg['init_obs'],
            'room_static_receps': ret_msg['room_static_receps'],
            'scene_name': scene_name,
            'task_ind': task_ind,
            'task_name' : task_json['task_name']
        }
        
        # 
        objects = self.env.last_event.metadata['objects']
        for o in objects:
            obj_id = o['objectId']
            obj_class = obj_id.split('|')[0]
        return init_obs
    
    @staticmethod
    def _yaw_to_dirs(y_deg: float):
        rad = math.radians(y_deg)
        fwd = (math.sin(rad), 0.0, math.cos(rad))
        right = (math.cos(rad), 0.0, -math.sin(rad))
        return fwd, right

    @staticmethod
    def _add_vec(a, b):
        return {"x": a["x"] + b["x"], "y": a["y"] + b["y"], "z": a["z"] + b["z"]}

    @staticmethod
    def _scale_vec(v, s):
        return {"x": v["x"] * s, "y": v["y"] * s, "z": v["z"] * s}

    @staticmethod
    def _vec(x, y, z):
        return {"x": x, "y": y, "z": z}
    
    def compute_agent_view_camera_back(self, controller, back=1.7, up=1.15, right_off=0.25, fov=60, ortho=False, ortho_size=1.2):
        md = controller.last_event.metadata
        agent_pos = md["agent"]["position"]
        agent_rot_y = md["agent"]["rotation"]["y"]
        fwd, right = self._yaw_to_dirs(agent_rot_y)
        fwd_v   = self._vec(fwd[0], 0.0, fwd[2])
        right_v = self._vec(right[0], 0.0, right[2])

        cam_pos = self._add_vec(agent_pos, self._add_vec(self._scale_vec(right_v, right_off), self._scale_vec(fwd_v, -back)))
        cam_pos["y"] += up

        dx = agent_pos["x"] - cam_pos["x"]
        dy = agent_pos["y"] - cam_pos["y"]
        dz = agent_pos["z"] - cam_pos["z"]
        yaw = math.degrees(math.atan2(dx, dz))
        dist_xz = math.hypot(dx, dz)
        pitch = -math.degrees(math.atan2(dy, dist_xz))

        props = {
            "position": cam_pos,
            "rotation": {"x": pitch, "y": yaw, "z": 0.0},
            "fieldOfView": float(fov),
            "orthographic": bool(ortho)
        }
        if ortho:
            props["orthographicSize"] = float(ortho_size)
        return props

    def update_camera_viewpoint(self, option="agent_view"):
        """
        Get the top-view frame of the current scene.
        This function is not working properly due to the issue of the ai2thor sometimes
        But it can be used for the visualization  

        Args:
            option (str): "map_view", "agent_view", "top_right", "top_left", "bottom_right", "bottom_left"
            Top views --> followed by the z-x plane bounding box 
        """
        if option=="map_view":
            event = self.env.step(
                action="GetMapViewCameraProperties"
            )
            camera_property = copy.deepcopy(
                event.metadata["actionReturn"]
            )
        elif option=="agent_view":
            # in case of the single camera property
            camera_property = self.compute_agent_view_camera_back(
                self.env, 
                back=1.7, 
                up=1.15, 
                right_off=0.25, 
                fov=60, 
                ortho=False
            )

        event = self.camera_event = self.env.step(
            action = "AddThirdPartyCamera",
            **camera_property
        )
        last_camera_frames = event.third_party_camera_frames
        if isinstance(last_camera_frames, list):
            for e in last_camera_frames:
                # update top view image to visualizer 
                e = cv2.cvtColor(e, cv2.COLOR_BGR2RGB)
                if option == "agent_view":
                    if self.show_agent_view:
                        self.visualizer.show(0, e)
                elif option == "map_view":
                    if self.show_top_view:
                        self.visualizer.show(1, e)
            return last_camera_frames[-1]
        else:
            last_camera_frames = cv2.cvtColor(last_camera_frames, cv2.COLOR_BGR2RGB)
            if option == "agent_view":
                if self.show_agent_view:
                    self.visualizer.show(0, e)
            elif option == "map_view":
                if self.show_top_view:
                    self.visualizer.show(1, e)
            return last_camera_frames

    def set_name_dict(self):
        last_event = self.env.last_event.metadata
        objects = last_event['objects']
        ignore_classes = ["Floor"]
        
        self.obj_id_to_high_name_dict = {}
        self.obj_name_to_high_name_dict = {}
        self.high_name_to_obj_id_dict = {}
        self.high_name_to_obj_name_dict = {}
        
        self.name_counter = {}  # 클래스별 이름 카운트

        # 객체 처리
        self.receptacles = []
        for obj in objects:
            
            obj_id = obj['objectId']
            obj_name = obj['name']
            obj_class_name = obj_id.split('|')[0]
            if obj_id.split("|")[-1] == "SinkBasin":
                obj_class_name = "SinkBasin"

            if "StoveBurner" in obj_class_name:
                self.receptacles.append(obj_id)
                continue
            if obj_class_name in ignore_classes:
                continue
            
            # class counter 
            if obj_class_name not in self.name_counter.keys():
                self.name_counter[obj_class_name] = 0
            
            self.name_counter[obj_class_name] += 1
            high_name = obj_class_name + " ({})".format(self.name_counter[obj_class_name])

            if "StoveKnob" in high_name:
                # get control objects 
                # corner case for StoveKnob - StoveBurner number matching...
                control_obj = obj['controlledObjects']
                for co_obj in control_obj:
                    co_obj_class_name = co_obj.split('|')[0]
                    if co_obj_class_name in ignore_classes:
                        continue
                    if co_obj_class_name not in self.name_counter.keys():
                        self.name_counter[co_obj_class_name] = 0
                    self.name_counter[co_obj_class_name] += 1
                    co_high_name = co_obj_class_name + " ({})".format(self.name_counter[co_obj_class_name])
                    self.obj_id_to_high_name_dict[co_obj] = co_high_name
                    self.obj_name_to_high_name_dict[co_obj_class_name] = co_high_name
                    self.high_name_to_obj_id_dict[co_high_name] = co_obj
                    self.high_name_to_obj_name_dict[co_high_name] = co_obj_class_name

            self.obj_id_to_high_name_dict[obj_id] = high_name
            self.obj_name_to_high_name_dict[obj_name] = high_name
            self.high_name_to_obj_id_dict[high_name] = obj_id
            self.high_name_to_obj_name_dict[high_name] = obj_name

            if obj['receptacle']:
                obj_class = obj['objectId'].split('|')[0]
                if obj_class in utils.STATIC_RECEPTACLES:
                    # print(f"--> Receptacle found: {high_name}, {obj_id}")
                    self.receptacles.append(obj_id)

    def restore_scene(self):
        self.reachable_positions, self.reachable_position_kdtree = self.get_reachable_positions()
        self.cur_receptacle = None

    def get_obj_idx(self, obj_id):
        
        for idx, obj in enumerate(self.env.last_event.metadata['objects']):

            if obj['objectId'] == obj_id:
                return idx

    def get_obj_information(self, obj_id):

        for obj in self.env.last_event.metadata['objects']:
            if obj['objectId'] == obj_id:
                return obj

    def get_reachable_positions(self):
        """
        Get reachable positions in the current scene.
        Returns:
            np.array: Array of reachable positions.
            spatial.KDTree: KDTree of reachable positions.
        """
        free_positions = self.env.step(action="GetReachablePositions").metadata["actionReturn"]
        free_positions = np.array([[p['x'], p['y'], p['z']] for p in free_positions])
        kd_tree = spatial.KDTree(free_positions)
        return free_positions, kd_tree

    def natural_word_to_ithor_name(self, w):
        # e.g., floor lamp -> FloorLamp
        if w == 'CD':
            return w
        else:
            return ''.join([string.capwords(x) for x in w.split()])
    
    def extract_number_from_string(self, s):
        match = re.match(r'^(.*\D)\s*(\d+)?$', s)
        if match:
            text_part = match.group(1).strip()
            number_part = int(match.group(2)) if match.group(2) else None
            return text_part, number_part
        else:
            return s, None
        
    def split_string_for_fill(self, s):
        # 将字符串按空格分割
        parts = s.split()
        
        # 找到倒数第二个部分的位置，并组合前面的部分为 part1，最后一个部分为 part2
        part1 = " ".join(parts[:-1])
        part2 = parts[-1]
        
        return part1, part2

    def get_visible_obj_nl_names_from_last_event(self):
        """
        Get nlnames of the visible object list from 
        Args:
        Returns:
            list: List of visible object names
        """
        ignore_classes = ["Floor"]
        visible_objects = []
        metadata = self.env.last_event.metadata
        objects = metadata['objects']
        for o in objects:
            # 1. check is visible
            obj_class = o['objectId'].split('|')[0]
            if obj_class in ignore_classes:
                continue
            # print(self.obj_id_to_high_name_dict[o['objectId']], o['visible'])
            if o['visible'] is True:
                obj_class_name = o['objectId'].split('|')[0]
                if obj_class_name in ignore_classes:
                    continue
                visible_objects.append(self.obj_id_to_high_name_dict[o['objectId']])
                # print("--> added.")

        return visible_objects

    def get_visual_obs_message(self):
        known_static_receps = self.receptacles
        known_static_receps = [self.obj_id_to_high_name_dict[x] for x in known_static_receps]

        # add visible objects feedback message 
        visible_objects = self.get_visible_obj_nl_names_from_last_event()
        
        vis_objs = list(set(visible_objects))
        vis_objs.sort()
        known_static_receps = list(set(known_static_receps))
        known_static_receps.sort()
        
        # group (1), (2), (3), .. (8) --> (1-8)
        final_vis_objs = utils.group_objects_by_name(vis_objs)
        final_known_static_receps = utils.group_objects_by_name(known_static_receps)

        # init obs str build
        static_obs_str = ', '.join(final_known_static_receps)
        vis_obs_str = ', '.join(final_vis_objs)
        return vis_obs_str, static_obs_str, final_vis_objs, final_known_static_receps
        
    def get_visual_obs_message_old(self):
        vis_objs = self.receptacles
        vis_objs = [self.obj_id_to_high_name_dict[x] for x in vis_objs]

        # print("------ visible objects ------")
        # for o in vis_objs:
        #     print(o)
        
        # add visible objects feedback message 
        visible_objects = self.get_visible_obj_nl_names_from_last_event()
        vis_objs += visible_objects
        vis_objs = list(set(vis_objs))

        # group (1), (2), (3), .. (8) --> (1-8)
        vis_objs.sort()
        final_vis_objs = utils.group_objects_by_name(vis_objs)
        # print("------ final visible objects ------")
        # for o in final_vis_objs:
        #     print(o)

        # init obs str build
        init_obs_str = ', '.join(final_vis_objs)
        ret = init_obs_str + '.'
        return ret, vis_objs

    def get_holding_obs_message(self):
        ret = None
        if len(self.env.last_event.metadata['inventoryObjects']) == 0:
            # I am not holding anything
            ret = 'I am not holding anything now. '
        else:
            # Add holding object feedback message 
            holding_obj_id = self.env.last_event.metadata['inventoryObjects'][0]['objectId']
            holding_obj = self.obj_id_to_high_name_dict[holding_obj_id]
            if isinstance(holding_obj, list):
                holding_obj_str = ', '.join(holding_obj)
            else:
                holding_obj_str = holding_obj
            ret = f'I am holding {holding_obj_str} now. '
        return ret

    def get_cooking_obs_message(self):
        # 1. get visible objects 
        ret = None
        latest_objects_states = self.env.last_event.metadata['objects']
        cooked_objs = []

        for o in latest_objects_states:
            obj_id = o['objectId']
            if o['visible'] is False:
                continue
            
            # 2. check if the object is cooking related 
            if o['cookable'] and o['isCooked']:
                cooked_objs.append(self.obj_id_to_high_name_dict[obj_id])

        if not len(cooked_objs) == 0:
            if ret is None:
                ret = ''
            cooked_objs.sort()
            cooked_objs_str_grouped = utils.group_objects_by_name(cooked_objs)
            cooked_obj_list_str = ', '.join(cooked_objs_str_grouped)
            if len(cooked_objs) == 1:
                ret += f'{cooked_obj_list_str} is cooked. '
            else:
                ret += f'{cooked_obj_list_str} are cooked. '

        return ret

    def get_temperature_obs_message(self):
        # 1. get visible objects 
        ignore_classes = ["Floor"]
        ret = None
        latest_objects_states = self.env.last_event.metadata['objects']
        hot_objs = []
        cold_objs = []

        for o in latest_objects_states:
            obj_id = o['objectId']
            obj_class = obj_id.split('|')[0]
            if obj_class in ignore_classes:
                continue
            if o['visible'] is False:
                continue
        
            temperature = o['temperature']
            if temperature == "Hot":
                hot_objs.append(self.obj_id_to_high_name_dict[obj_id])
            elif temperature == "Cold":
                cold_objs.append(self.obj_id_to_high_name_dict[obj_id])

        if not len(hot_objs) == 0:
            if ret is None:
                ret = ''
            hot_objs.sort()
            hot_objs_str_grouped = utils.group_objects_by_name(hot_objs)
            hot_obj_list_str = ', '.join(hot_objs_str_grouped)
            if len(hot_objs) == 1:
                ret += f'{hot_obj_list_str} is hot. '
            else:
                ret += f'{hot_obj_list_str} are hot. '
        
        if not len(cold_objs) == 0:
            if ret is None:
                ret = ''
            cold_objs.sort()
            cold_objs_str_grouped = utils.group_objects_by_name(cold_objs)
            cold_obj_list_str = ', '.join(cold_objs_str_grouped)
            if len(cold_objs) == 1:
                ret += f'{cold_obj_list_str} is cold. '
            else:
                ret += f'{cold_obj_list_str} are cold. '

        return ret

    def get_filling_obs_message(self):
        ignore_classes = ["Floor"]
        latest_objects_states = self.env.last_event.metadata['objects']
        filled_objs = []
        empty_objs = []
        ret = None

        for o in latest_objects_states:
            obj_id = o['objectId']
            can_fill = o['canFillWithLiquid']
            is_fill = o['isFilledWithLiquid']
            # print(o.keys())
            obj_class = obj_id.split('|')[0]
            if obj_class in ignore_classes:
                continue
            # print(self.obj_id_to_high_name_dict[obj_id], can_fill, o['visible'], is_fill)
            if not is_fill:
                continue

            if o['visible'] is False:
                continue
            liquid_type = o['fillLiquid']
            if is_fill:
                filled_objs.append(
                    {
                        'name': self.obj_id_to_high_name_dict[obj_id],
                        'liquid_type': liquid_type,
                    }
                )
                    
        if not len(filled_objs) == 0:
            if ret is None:
                ret = ''

            filled_obj_names = [x['name'] for x in filled_objs]
            # filled_obj_liquids = [x['liquid_type'] for x in filled_objs]
            
            water_filled_objs = []
            coffee_filled_objs = []
            wine_filled_objs = []

            for i, filled_obj in enumerate(filled_objs):
                print(filled_obj['liquid_type'])
                if filled_obj['liquid_type'] == 'water' or filled_obj['liquid_type'] == 'Water':
                    water_filled_objs.append(filled_obj_names[i])
                elif filled_obj['liquid_type'] == 'coffee' or filled_obj['liquid_type'] == 'Coffee':
                    coffee_filled_objs.append(filled_obj_names[i])
                elif filled_obj['liquid_type'] == 'wine' or filled_obj['liquid_type'] == 'Wine':
                    wine_filled_objs.append(filled_obj_names[i])

            water_filled_objs_str_grouped = utils.group_objects_by_name(water_filled_objs)
            water_filled_obj_list_str = ', '.join(water_filled_objs_str_grouped)
            # print("water_filled_objs_str_grouped: ", water_filled_objs_str_grouped)
            # print("water_filled_obj_list_str: ", water_filled_obj_list_str)
            # print(len(water_filled_objs))
            coffee_filled_objs_str_grouped = utils.group_objects_by_name(coffee_filled_objs)
            coffee_filled_obj_list_str = ', '.join(coffee_filled_objs_str_grouped)
            # print("coffee_filled_objs_str_grouped: ", coffee_filled_objs_str_grouped)
            # print("coffee_filled_obj_list_str: ", coffee_filled_obj_list_str)
            # print(len(coffee_filled_objs))
            wine_filled_objs_str_grouped = utils.group_objects_by_name(wine_filled_objs)
            wine_filled_obj_list_str = ', '.join(wine_filled_objs_str_grouped)
            # print("wine_filled_objs_str_grouped: ", wine_filled_objs_str_grouped)
            # print("wine_filled_obj_list_str: ", wine_filled_obj_list_str)
            # print(len(wine_filled_objs))
            
            if not len(water_filled_objs) == 0:
                if len(water_filled_objs) == 1:
                    ret += f'{water_filled_obj_list_str} is filled with water. '
                else:
                    ret += f'{water_filled_obj_list_str} are filled with water. '
            
            if not len(coffee_filled_objs) == 0:
                if len(coffee_filled_objs) == 1:
                    ret += f'{coffee_filled_obj_list_str} is filled with coffee. '
                else:
                    ret += f'{coffee_filled_obj_list_str} are filled with coffee. '
            if not len(wine_filled_objs) == 0:    
                if len(wine_filled_objs) == 1:
                    ret += f'{wine_filled_obj_list_str} is filled with wine. '
                else:
                    ret += f'{wine_filled_obj_list_str} are filled with wine. '

        return ret        

    def get_dirty_obs_message(self):
        latest_objects_states = self.env.last_event.metadata['objects']
        dirty_objs = []
        clean_objs = []
        ret = None

        for o in latest_objects_states:
            obj_id = o['objectId']
            
            if o['visible'] is False:
                continue

            if o['dirtyable'] and o['isDirty']:
                dirty_objs.append(self.obj_id_to_high_name_dict[obj_id])
            elif o['dirtyable'] and not o['isDirty']:
                clean_objs.append(self.obj_id_to_high_name_dict[obj_id])

        if not len(dirty_objs) == 0:
            if ret is None:
                ret = ''
            dirty_objs.sort()
            dirty_objs_str_grouped = utils.group_objects_by_name(dirty_objs)
            dirty_obj_list_str = ', '.join(dirty_objs_str_grouped)
            if len(dirty_objs) == 1:
                ret += f'{dirty_obj_list_str} is dirty. '
            else:
                ret += f'{dirty_obj_list_str} are dirty. '

        if not len(clean_objs) == 0:
            if ret is None:
                ret = ''
            clean_objs.sort()
            clean_objs_str_grouped = utils.group_objects_by_name(clean_objs)
            clean_obj_list_str = ', '.join(clean_objs_str_grouped)
            if len(clean_objs) == 1:
                ret += f'{clean_obj_list_str} is clean. '
            else:
                ret += f'{clean_obj_list_str} are clean. '

        return ret        

    def get_open_obs_message(self):
        latest_objects_states = self.env.last_event.metadata['objects']
        opened_objs = []
        closed_objs = []
        ret = None

        for o in latest_objects_states:
            obj_id = o['objectId']
            
            if o['visible'] is False:
                continue
            if o['openable'] and o['isOpen']:
                opened_objs.append(self.obj_id_to_high_name_dict[obj_id])
            elif o['openable'] and not o['isOpen']:
                closed_objs.append(self.obj_id_to_high_name_dict[obj_id])
        
        if not len(opened_objs) == 0:
            if ret is None:
                ret = ''
            
            opened_objs.sort()
            opened_objs_str_grouped = utils.group_objects_by_name(opened_objs)
            opened_obj_list_str = ', '.join(opened_objs_str_grouped)
            if len(opened_objs) == 1:
                ret += f'{opened_obj_list_str} is opened. '
            else:
                ret += f'{opened_obj_list_str} are opened. '

        if not len(closed_objs) == 0:
            if ret is None:
                ret = ''
            closed_objs.sort()
            closed_objs_str_grouped = utils.group_objects_by_name(closed_objs)
            closed_obj_list_str = ', '.join(closed_objs_str_grouped)
            if len(closed_objs) == 1:
                ret += f'{closed_obj_list_str} is closed. '
            else:
                ret += f'{closed_obj_list_str} are closed. '

        return ret        

    def get_toggle_obs_message(self):
        latest_objects_states = self.env.last_event.metadata['objects']
        turned_on_objs = []
        turned_off_objs = []
        ret = None
        for o in latest_objects_states:
            obj_id = o['objectId']
            
            if o['visible'] is False:
                continue
            
            if o['toggleable'] and o['isToggled']:
                turned_on_objs.append(self.obj_id_to_high_name_dict[obj_id])
            elif o['toggleable'] and not o['isToggled']:
                turned_off_objs.append(self.obj_id_to_high_name_dict[obj_id])

        if not len(turned_on_objs) == 0:
            if ret is None:
                ret = ''
            
            turned_on_objs.sort()
            turned_on_objs_str_grouped = utils.group_objects_by_name(turned_on_objs)
            turned_obj_list_str = ', '.join(turned_on_objs_str_grouped)
            if len(turned_on_objs) == 1:
                ret = f'{turned_obj_list_str} is turned on. '
            else:
                ret = f'{turned_obj_list_str} are turned on. '

        if not len(turned_off_objs) == 0:
            if ret is None:
                ret = ''
            
            turned_off_objs.sort()
            turned_off_objs_str_grouped = utils.group_objects_by_name(turned_off_objs)
            turned_off_obj_list_str = ', '.join(turned_off_objs_str_grouped)
            if len(turned_off_objs) == 1:
                ret = f'{turned_off_obj_list_str} is turned off. '
            else:
                ret = f'{turned_off_obj_list_str} are turned off. '

        return ret        

    def pick_obj(self, target_nl_name):
        """
        Pick up the target object wrapper function

        Args : 
            target_nl_name (str): Target object name in natural language.
        
        Returns:
            str: Pick up result message.
        """
        ret_msg = ''
        action_success = False

        # 1. check the target object is visible 
        visible_objects = self.get_visible_obj_nl_names_from_last_event()
        if target_nl_name not in visible_objects:
            ret_msg = f'Failed to pick up {target_nl_name}. The {target_nl_name} is not visible. '
        else:
            # 2. pick up the target object
            # ret_msg = self._pick_obj_5(target_nl_name) <-- not working... 
            ret_msg, action_success = self._pick_obj(target_nl_name)
        
        return ret_msg, action_success

    def update_camera(self):
        """Do meaningless actions for updating the screen"""
        self.env.step(action="LookUp",degrees=1)
        self.env.step(action="LookDown",degrees=1)

    def get_parent_recep(self, target_obj_nl_name):
        if not target_obj_nl_name in self.high_name_to_obj_id_dict.keys():
            return
        target_obj_id = self.high_name_to_obj_id_dict[target_obj_nl_name]

        # if not receptacle, update the receptacle that the object is in
        metadata = self.env.last_event.metadata['objects']
        parent_recep_nlname = None
        for o in metadata:
            if o['objectId'] == target_obj_id:
                # get parent object id 
                if o['parentReceptacles'] is None:# mostly on the floor after the drop action
                    # update to floor
                    return "unknown"

                parent_recep_id = o['parentReceptacles'][-1] 
                if parent_recep_id in self.obj_id_to_high_name_dict.keys():
                    parent_recep_nlname = self.obj_id_to_high_name_dict[parent_recep_id]
                else:
                    if parent_recep_id == "Floor":
                        self.cur_receptacle = target_obj_nl_name
                        return
                    else:
                        parent_recep_nlname = None # don't know the parent receptacle
                self.cur_receptacle = parent_recep_nlname
        return parent_recep_nlname

    def update_last_visited_receptacles(self, target_obj_nl_name):
        """
        Update latest visited receptacle by target object name in natural language.
        If the target object in pre-acquired static receptacle, update the last visited receptacle.
        Else if the parent of the target object is floor 
        then update the last visited receptacle as the target object 
        Because the target object is on the floor = static receptacle 
            --> rarly case, if the object is dropped on the floor, 
                might be cause the error, but it is not a big problem
                due to the drop aciton is not allowed in the current task setting.
        Args:
            target_obj_nl_name (str): Target object name in natural language

        Returns:
            None
        """
        target_obj_class_name = target_obj_nl_name.split(" ")[0]
        if not target_obj_nl_name in self.high_name_to_obj_id_dict.keys():
            return

        target_obj_id = self.high_name_to_obj_id_dict[target_obj_nl_name]
        # 1. check is receptacle or not 
        # if not self.is_holodeck:
        if target_obj_class_name in utils.ALFRED_RECEP :
            # static recep + movable recep, due to put obj to movable recep 
            # if receptacle, update the last visited receptacle
            self.cur_receptacle = target_obj_nl_name
        else:
            # if not receptacle, update the receptacle that the object is in
            metadata = self.env.last_event.metadata['objects']
            for o in metadata:
                if o['objectId'] == target_obj_id:
                    # get parent object id 
                    if o['parentReceptacles'] is None:# mostly on the floor after the drop action
                        # update to floor
                        self.cur_receptacle = "Floor"
                        return

                    parent_recep_id = o['parentReceptacles'][-1] 
                    if parent_recep_id in self.high_name_to_obj_id_dict.keys():
                        parent_recep_nlname = self.high_name_to_obj_id_dict[parent_recep_id]
                    else:
                        if parent_recep_id == "Floor":
                            self.cur_receptacle = target_obj_nl_name
                            return
                        else:
                            parent_recep_nlname = None # don't know the parent receptacle
                    self.cur_receptacle = parent_recep_nlname
        
    def _pick_obj(self, target_nl_name):
        """
        Pick up the target object by nl name

        Args:
            target_nl_name (str): Target object name in natural language.
        
        Returns:
            str: Pick up result message.
        """
        action_success=False
        ret_msg = ''
        obj_id = self.high_name_to_obj_id_dict[target_nl_name]

        # if pick object, then the agent should close to the any static receptacle 
        # that contain the target object 
        # when the agent directly pick up object after room to room navigation,
        # this might be helpful to update the last visited receptacle
        #self.update_last_visited_receptacles(target_nl_name)
        # parent = self.get_parent_recep(target_nl_name)
        self.env.step(action="UnpausePhysicsAutoSim")
        for j in range(16):
            if j == 1:
                self.env.step(dict(action="LookUp", degrees=15))
            elif j == 2:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 3:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 4:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 5:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 6:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 7:
                self.env.step(dict(action="LookDown"), degrees=55)
            elif j == 8:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 9:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 10:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 11:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 12:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 13:
                self.env.step(dict(action="LookUp"), degrees=40)
                self.env.step(dict(action="LookUp"))
                self.env.step(dict(action="LookUp"))
                self.env.step(dict(action="MoveBack"))
            elif j == 14:
                self.env.step(dict(action="MoveAhead"))
                for r in range(4):
                    self.env.step(dict(action="MoveRight"))
            elif j == 15:
                for r in range(8):
                    self.env.step(dict(action="MoveLeft"))
            elif j == 16:
                for r in range(4):
                    self.env.step(dict(action="MoveRight"))
                # this somehow make putobject success in some cases
                self.env.step(dict(  
                    action="RotateRight",
                    degrees=15
                ))

            self.env.step(
                action="PickupObject",
                objectId=obj_id,
                forceAction=True,
                manualInteract=True, 
            )

            if not self.env.last_event.metadata['lastActionSuccess']:
                log.warning(
                    f"PickupObject action failed, the error message is {self.env.last_event.metadata['errorMessage']}"
                )
                ret_msg = f'Picking up {target_nl_name} failed. '
            else:
                ret_msg = f'You pick up {target_nl_name}. '
                # if parent is not None and parent != "unknown":
                #     self.cur_receptacle =  parent
                action_success=True
                break

        self.env.step(action="PausePhysicsAutoSim")
        self.update_camera()
        return ret_msg, action_success

    def put_down(self, target_nl_name):
        """
        Put down the target object wrapper function

        Args :
            target_nl_name (str): Target object name in natural language.
        
        Returns:
            str: Put down result message.
        """
        ret_msg = ''
        action_success = False
        # 1. check the object is in the inventory 
        if len(self.env.last_event.metadata['inventoryObjects']) == 0:
            ret_msg = f'Nothing Done. Robot is not holding any object. '
            return ret_msg, action_success
        else:
            # 2. check holding object and target object is the same
            if target_nl_name not in self.high_name_to_obj_id_dict.keys():
                return f'Nothing Done. Robot is not holding {target_nl_name}. ', action_success
            obj_id = self.high_name_to_obj_id_dict[target_nl_name]
            holding_obj_id = self.env.last_event.metadata['inventoryObjects'][0]['objectId']
            if obj_id != holding_obj_id:
                ret_msg = f'Nothing Done. Robot is not holding {target_nl_name}. '
                return ret_msg, action_success
        
        # 3. try to put down 
        recep_nl_name = self.cur_receptacle
        if recep_nl_name == "Floor":
            # modtly on the floor after the drop action
            ret_msg = f'Cannot find a place to put {target_nl_name}. '
            return ret_msg, action_success
        if recep_nl_name is None:
            # edge case if directly go to the room and put down
            ret_msg = f'You can not put down {target_nl_name} here. You need to go to the receptacle first. '
            return ret_msg, action_success

        recep_id = self.high_name_to_obj_id_dict[recep_nl_name]
        self.env.step(action="UnpausePhysicsAutoSim")
        put_task_done=False
        action_success=False
        holding_obj_info = self.get_obj_information(holding_obj_id)

        log.info(f'put {target_nl_name} on {recep_nl_name}')

        for j in range(18):  # move/look around or rotate obj

            if not recep_id:
                ret_msg = f'Cannot find {recep_nl_name}. '
                continue

            ### Try To change camera 
            # look up (put action fails when a receptacle is not visible)
            if j == 1:
                self.env.step(dict(action="LookUp", degrees=15))
            elif j == 2:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 3:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 4:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 5:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 6:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 7:
                self.env.step(dict(action="LookDown"), degrees=55)
            elif j == 8:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 9:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 10:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 11:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 12:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 13:
                self.env.step(dict(action="LookUp"), degrees=40)
                self.env.step(dict(action="LookUp"))
                self.env.step(dict(action="LookUp"))
                self.env.step(dict(action="MoveBack"))
            elif j == 14:
                self.env.step(dict(action="MoveAhead"))
                for r in range(4):
                    self.env.step(dict(action="MoveRight"))
            elif j == 15:
                for r in range(8):
                    self.env.step(dict(action="MoveLeft"))
            elif j == 16:
                for r in range(4):
                    self.env.step(dict(action="MoveRight"))
                self.env.step(dict(  # this somehow make putobject success in some cases
                    action="RotateRight",
                    degrees=15
                ))

            ###### Put Down Action 
            self.env.step(action="UnpausePhysicsAutoSim")
            self.env.step(dict(
                action="PutObject",
                objectId=recep_id,
                forceAction=True
            ))
            putobject_success = self.env.last_event.metadata['lastActionSuccess']
            self.env.step(action="PausePhysicsAutoSim")

            if not putobject_success:
                if j == 17:
                    # self.controller.step(dict(
                    #     action="LookAtObjectCenter",
                    #     objectId=recep_id
                    # ))
                    # self.update_camera()
                    # for o in self.controller.last_event.metadata['objects']:
                    #     if o['objectId'] == recep_id:
                    #         obj_pos = o['position']
                    # rot_angle = self.get_camera_rotation_angle_to_point(
                    #     self.controller.last_event.metadata['agent']['rotation'],
                    #     self.controller.last_event.metadata['agent']['position'],
                    #     obj_pos
                    # # )
                    # self.rotate_camera_by_angle_right(rot_angle)
                    # self.update_camera()
                    event = self.env.step(
                        action="GetSpawnCoordinatesAboveReceptacle",
                        objectId=recep_id,
                        anywhere=False
                    )
                    position_above = event.metadata['actionReturn']
                    if position_above is not None:
                        if len(position_above) == 0:
                            event = self.env.step(
                                action="GetSpawnCoordinatesAboveReceptacle",
                                objectId=recep_id,
                                anywhere=True
                            )
                            position_above = event.metadata['actionReturn']
                            if len(position_above) == 0:
                                ret_msg = f'Cannot find a place to put {target_nl_name}. '
                                action_success=False
                                break
                    else:
                        ret_msg = f'Cannot find a place to put {target_nl_name}. '
                        action_success=False
                        break

                    self.env.step(action="UnpausePhysicsAutoSim")
                    self.env.step(
                        action="PlaceObjectAtPoint",
                        objectId=holding_obj_id,
                        position = {
                            "x": sum([tmp['x'] for tmp in position_above])/len(position_above),
                            "y": sum([tmp['y'] for tmp in position_above])/len(position_above),
                            "z": sum([tmp['z'] for tmp in position_above])/len(position_above)
                        },
                    )
                    action_success = self.env.last_event.metadata['lastActionSuccess']
                    print(">>>>>> PlaceObjectAtPoint action success: ", action_success)
                    self.env.step(action="PausePhysicsAutoSim")
                    break
                else:
                    ret_msg = f'Putting the object on {recep_nl_name} failed'
                    action_success=False
            else:
                action_success=True
                break
                
        #### In case of all actions are done
        if action_success:
            # 1. get latest holding object information 
            latest_objects = self.env.last_event.metadata['objects']
            for _o in latest_objects:
                if holding_obj_info['objectId'] == _o['objectId']:
                    holding_obj_info = _o
                    break
            if not holding_obj_info['parentReceptacles'] is None:
                if recep_id in holding_obj_info['parentReceptacles']:
                    put_task_done=True
                else:
                    log.warning(f"PutObject action failed: {self.env.last_event.metadata['errorMessage']}, trying again...")
                    ret_msg = f'Putting the object on {recep_nl_name} failed. '
                    put_task_done=False
                    if "Sliced" in target_nl_name and "Toaster" in recep_nl_name:
                        ret_msg = f"Putting the object on {recep_nl_name} failed. The slice too large to put into the toaster. "
            else:
                log.warning(f"PutObject action is successed. But can not find the parent receptacle after putting down. \nThis might be caused by the physics interaction problem when place object on the target receptacle in last try: {self.env.last_event.metadata['errorMessage']}")
                ret_msg = f'Putting the object on {recep_nl_name} failed. '
                put_task_done=False
                if "Sliced" in target_nl_name and "Toaster" in recep_nl_name:
                    ret_msg = f"Putting the object on {recep_nl_name} failed. The slice too large to put into the toaster. "
        else:
            # 2. if put down failed, then update the last visited receptacle as the target object
            put_task_done=False
            ret_msg = f'Putting the object on {recep_nl_name} failed. '
            if "Sliced" in target_nl_name and "Toaster" in recep_nl_name:
                ret_msg = f"Putting the object on {recep_nl_name} failed. The slice too large to put into the toaster. "

        if put_task_done:
            ret_msg = f'You put down {target_nl_name} on {recep_nl_name}.'
        
        # print("action sucess ", put_task_done)
        # print("action result : ", ret_msg)
        # print(self.controller.last_event.metadata['errorMessage'])
        # print("last action success ", self.controller.last_event.metadata['lastActionSuccess'])
        self.update_camera()
        self.env.step(action="PausePhysicsAutoSim")
        return ret_msg, put_task_done
   
    def llm_skill_interact(self, instruction= None):
        obs_obj = None
        final_feedback_message = ""
        init_obs_message = None
        ret=None

        if instruction is None:
            # get init_observation 
            # visual_obs_message, obs_obj = self.get_visual_obs_message()
            visual_obs_message, static_recep_message, final_vis_objs, final_static_receps = self.get_visual_obs_message()
            
            # return vis_obs_str, static_obs_str, final_vis_objs, final_known_static_receps
            obs_obj = final_vis_objs + final_static_receps
            obs_obj = list(set(obs_obj))
            if len(final_vis_objs)== 0:
                ret_feedback_message = 'You are in the middle of a room. Looking quickly around you, you see appliances and furnitures: '+ static_recep_message 
            else:       
                ret_feedback_message = 'You are in the middle of a room. Looking quickly around you, you see appliances and furnitures: '+ static_recep_message + '. In front of you there are: '+visual_obs_message+ '. ' 
            action_success = True
            final_feedback_message = ret_feedback_message
            init_obs_message = final_feedback_message
        else:
            action_success=False
            if instruction.startswith("go to "):
                target_nl_name = instruction.replace('go to a ', '').replace('go to an ', '').replace('go to ', '')
                ret, action_success = self.go_to_obj(target_nl_name)
                self.update_last_visited_receptacles(target_nl_name)
                
            elif instruction.startswith("pick up "):
                target_nl_name= instruction.replace('pick up the ', '').replace('pick up ', '')
                ret, action_success = self.pick_obj(target_nl_name)

            elif instruction.startswith("put down "):
                m = re.match(r'put down (.+)', instruction)
                target_nl_name = m.group(1).replace('the ', '')
                ret, action_success = self.put_down(target_nl_name)
                
            elif instruction.startswith("open "):
                target_nl_name = instruction.replace('open the ', '').replace('open ', '')
                ret, action_success = self.open(target_nl_name)
                
            elif instruction.startswith("close "):
                target_nl_name = instruction.replace('close the ', '').replace('close ', '')
                ret, action_success = self.close(target_nl_name)
                
            elif instruction.startswith("turn on "):
                target_nl_name = instruction.replace('turn on the ', '').replace('turn on ', '')
                ret, action_success = self.turnon(target_nl_name)
                
            elif instruction.startswith("turn off "):
                target_nl_name = instruction.replace('turn off the ', '').replace('turn off ', '')
                ret, action_success = self.turnoff(target_nl_name)
                
            elif instruction.startswith("slice "):
                target_nl_name = instruction.replace('slice the ', '').replace('slice ', '')
                ret, action_success = self.slice(target_nl_name)

            elif instruction.startswith("drop"):
                ret, action_success = self.drop()

            elif instruction.startswith("throw"):
                target_nl_name = instruction.replace('throw the ', '').replace('throw ', '')
                ret, action_success = self.throw(target_nl_name)

            elif instruction.startswith("pour"):
                target_nl_name = instruction.replace('pour into the', '').replace('pour into', '')
                ret, action_success = self.pour(target_nl_name)

            elif instruction.startswith("break"):
                target_nl_name = instruction.replace('break the ', '').replace('break ', '')
                ret, action_success = self.breaking(target_nl_name)
            
            elif instruction.startswith("empty"):
                target_nl_name = instruction.replace('empty the ', '').replace('empty ', '')
                ret = self.empty(target_nl_name)
            
            elif instruction.startswith("showid"): # debug only!! 
                target_nl_name = instruction.replace('showid the ', '').replace('showid ', '')
                ret = ""
                action_success = True
                obj_info = self.get_obj_information(self.high_name_to_obj_id_dict[target_nl_name])
                position = obj_info['position']
                rotation = obj_info['rotation']
                if target_nl_name in self.high_name_to_obj_id_dict.keys():
                    print(f"Object ID of {target_nl_name} is {self.high_name_to_obj_id_dict[target_nl_name]}")
                    print(f"Object position of {target_nl_name} is {position}")
                    print(f"Object rotation of {target_nl_name} is {rotation}")

            if ret is None:
                log.warning(f"llm_skill_interact failed")
                log.warning(f"not supported instruction: {instruction}")
                final_feedback_message = f"Not supported instruction: {instruction}. "
            else:
                if not action_success:
                    log.warning(f"llm_skill_interact failed")
                    log.warning(f"errorMessage: {self.env.last_event.metadata['errorMessage']}")
                    log.warning(f"returned msg: {ret}")
                
                else:
                    log.info(f"Last action succeeded")
            
                final_feedback_message+=ret
        
            # add observation message 
            # visual_obs_message, obs_obj = self.get_visual_obs_message()
            # final_feedback_message+= 'You see: '+visual_obs_message
            visual_obs_message, static_recep_message, final_vis_objs, final_static_receps = self.get_visual_obs_message()
            
            # return vis_obs_str, static_obs_str, final_vis_objs, final_known_static_receps
            obs_obj = final_vis_objs + final_static_receps
            obs_obj = list(set(obs_obj))

            if len(final_vis_objs)== 0:
                ret_feedback_message = 'You have seen appliances and furnitures: '+ static_recep_message
            else:
                ret_feedback_message = 'You have seen appliances and furnitures: '+ static_recep_message + '. In front of you, there are: '+visual_obs_message+ '. '
            
            final_feedback_message += ret_feedback_message

        # add toggle feedback message
        toggle_obs_message = self.get_toggle_obs_message()
        if not toggle_obs_message is None:
            final_feedback_message += toggle_obs_message

        # add open feedback message
        open_obs_message = self.get_open_obs_message()
        if not open_obs_message is None:
            final_feedback_message += open_obs_message

        # add dirty feedback message 
        dirty_obs_message = self.get_dirty_obs_message()
        if not dirty_obs_message is None:
            final_feedback_message += dirty_obs_message
        
        # filling information 
        filling_obs_message = self.get_filling_obs_message()
        if not filling_obs_message is None:
            final_feedback_message += filling_obs_message
        
        # cooking information 
        cooking_obs_message = self.get_cooking_obs_message()
        if not cooking_obs_message is None:
            final_feedback_message += cooking_obs_message
        
        # temperature information 
        temperature_obs_message = self.get_temperature_obs_message()
        if not temperature_obs_message is None:
            final_feedback_message += temperature_obs_message


        # holding object feedback signal
        holding_obs_message = self.get_holding_obs_message()
        if not holding_obs_message is None:
            final_feedback_message += holding_obs_message 

        # update found objects 
        self.update_found_objects()    
        
        ret_dict = {
            'action': instruction,
            'success': action_success,
            'message': final_feedback_message,
            'nl_obs_partial_objs_info': obs_obj,  
            'init_obs': init_obs_message,
            'room_static_receps': [self.obj_id_to_high_name_dict[x] for x in self.receptacles],
        }
        self.get_third_party_camera_frames(instruction)

        # check online metric 

        if self.step_const_checker is not None:
            if self.step_const_checker.is_violated:
                log.warning(f"[Online Metric] Step constraint is violated. Task Failed.")
                # ret_dict['constraint_violation'] = self.step_const_checker.get_violation_message()
                print(self.step_const_checker.colors.RED + "[Online Metric] Step constraint is violated. Task Failed."+ self.step_const_checker.colors.RESET)
            else:
                # check triggering condition 
                self.step_const_checker.check_triggering_condition()

        # for debug only 
        if self.final_condition_checker is not None:
            self.final_condition_checker.check()

        if self.avoid_condition_checker is not None:
            if self.avoid_condition_checker.is_violated:
                log.warning(f"[Online Metric] Avoid condition is violated. Task Failed.")
                print(self.avoid_condition_checker.colors.RED + "[Online Metric] Avoid condition is violated. Task Failed." + self.avoid_condition_checker.colors.RESET)
            else:
                # check triggering condition
                self.avoid_condition_checker.check_triggering_condition()

        return ret_dict
    
    def get_goal_success(self):
        
        is_success_step_const = False
        state_condition_successes=[]
        num_positive = 0

        if self.final_condition_checker is not None:
            state_condition_successes = self.final_condition_checker.check()

        if self.step_const_checker is not None:
            is_success_step_const = self.step_const_checker.check_triggering_condition()

            if self.treat_step_constraint_as_goal:
                if self.step_const_checker.is_done:
                    state_condition_successes.append(True)
                else:
                    state_condition_successes.append(False)

        if self.avoid_condition_checker is not None:
            is_success_avoid_condition = self.avoid_condition_checker.check_triggering_condition()
            
        if self.step_const_checker is not None:
            if not is_success_step_const:
                print(self.step_const_checker.colors.RED + "[Final Metric] Step constraint is violated. Task Failed." + self.step_const_checker.colors.RESET)
                log.warning(f"[Final Metric] Step constraint is violated. Task Failed.")
        
        for state_condition_success in state_condition_successes:
            if state_condition_success:
                num_positive += 1
        
        sub_goal_success_rate = num_positive / len(state_condition_successes) if len(state_condition_successes) > 0 else 0.0
        if all(state_condition_successes):
            if self.step_const_checker is not None:
                print(self.step_const_checker.colors.GREEN + "[Final Metric] All sub-goals are achieved. Task Success." + self.step_const_checker.colors.RESET)
                log.info(f"[Final Metric] All sub-goals are achieved. Task Success.")
            elif self.avoid_condition_checker is not None:
                print(self.avoid_condition_checker.colors.GREEN + "[Final Metric] All sub-goals are achieved. Task Success." + self.avoid_condition_checker.colors.RESET)
                log.info(f"[Final Metric] All sub-goals are achieved. Task Success.")
            sub_goal_success_rate = 1.0
        else:
            if self.step_const_checker is not None:
                print(self.step_const_checker.colors.RED + "[Final Metric] Not all sub-goals are achieved. Task Failed." + self.step_const_checker.colors.RESET)
                log.warning(f"[Final Metric] Not all sub-goals are achieved. Task Failed.")
            elif self.avoid_condition_checker is not None:
                print(self.avoid_condition_checker.colors.RED + "[Final Metric] Not all sub-goals are achieved. Task Failed." + self.avoid_condition_checker.colors.RESET)
                log.warning(f"[Final Metric] Not all sub-goals are achieved. Task Failed.")
        
        return {
            'step_constraint_success': is_success_step_const,
            'sub_goal_success_rate': sub_goal_success_rate,
            'sub_goal_successes': state_condition_successes,
            'is_safety_constraints_success': is_success_step_const
        }

    def empty(self, target_nl_name):
        """
        Empty action wrapper function
        The agent will empty the holding object if it is fillable and filled with liquid.

        Args:
            target_nl_name (str): Target object name in natural language.

        Returns:
            str: Empty result message.
        """
        log.info(f'empty {target_nl_name}')
        ret_msg = ''

        if len(self.env.last_event.metadata['inventoryObjects']) == 0:
            return 'Nothing Done. Robot is not holding any object. '
        
        target_obj_id = self.high_name_to_obj_id_dict[target_nl_name]
        holding_obj_id = self.env.last_event.metadata['inventoryObjects'][0]['objectId']
        holding_obj_inf = self.get_obj_information(holding_obj_id)

        if "SinkBasin" in target_nl_name:
            if len(self.env.last_event.metadata['inventoryObjects']) > 1:
                return 'Nothing Done. You can not empty the object you are holding to the sink basin. '
            else:
                self.env.step(
                    action="EmptyLiquidFromObject",
                    objectId=holding_obj_id
                )
                if not self.env.last_event.metadata['lastActionSuccess']:
                    return 'Empty action failed. '
                else:
                    return f'The {target_nl_name} is empty now. '

        # check the input target nl name is same as the holding object
        if target_obj_id != holding_obj_id:
            return 'Nothing Done. Robot is not holding the target object. '

        # nonsense but just in case
        if holding_obj_id is None:
            return 'Nothing Done. Cannot find the object. '

        # check the holding object is fillable
        can_pour = holding_obj_inf['canFillWithLiquid']
        if not can_pour:
            return 'Nothing Done. You can not empty the object you are holding. '
        
        is_filled = holding_obj_inf['isFilledWithLiquid']

        if not is_filled:
            # object is not filled with liquid --> do nothing
            return 'Nothing Done. The object you holding is not filled with liquid. '
        else:
            # Do pouring motion : Don't care about the target object
            # just animate the pouring motion
            # Do animation for emptying the object
            is_dirty_liquid = False
            if 'fillLiquid' in holding_obj_inf.keys():
                liquid_type = holding_obj_inf['fillLiquid']
                is_dirty_liquid = True
            
            angel_list = [-30.0, -60.0, -90.0, 90.0, 60.0, 30.0]
            for angel in angel_list:
                self.env.step(
                    action="RotateHeldObject",
                    # pitch=angel
                    roll = angel
                )
                self.get_third_party_camera_frames()

            holding_obj_inf = self.get_obj_information(holding_obj_id)
            is_filled = holding_obj_inf['isFilledWithLiquid']

            # Change the object states 
            # 1. empty the holding object 

            if is_filled:
                self.env.step(
                    action="EmptyLiquidFromObject",
                    objectId=holding_obj_id
                )

            if not self.env.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Empty action failed. "
            else:
                ret_msg = f'The {target_nl_name} is empty now. '

            # if is_dirty_liquid:
            #     if liquid_type == "coffee" or liquid_type =="wine":
            #         # make the object dirty 
            #         self.env.step(
            #             action="DirtyObject",
            #             objectId=holding_obj_id
            #         )
          
        return ret_msg

    def breaking(self, target_nl_name):
        """
        Break action wrapper function

        Args:
            target_nl_name (str): Target object name in natural language.
        
        Returns:
            str: Break result message.
        """
        ignore_classes =["Floor"]
        action_success=False
        ret = ''

        # check holding object and target object is the same
        target_obj_id = self.high_name_to_obj_id_dict[target_nl_name]
        
        # check the target object is breakable
        objects = self.env.last_event.metadata['objects']
        for o in objects:
            if o['objectId'] == target_obj_id:
                if not o['breakable']:
                    return f'The {target_nl_name} cannot be broken. ', action_success
                else:
                            
                    self.last_event = self.env.step(
                        action="BreakObject",
                        objectId=target_obj_id
                    )
                    
                    if not self.env.last_event.metadata['lastActionSuccess']:
                        ret = f'The {target_nl_name} break failed. '
                    else:
                        action_success=True
                        ret =  f'You break the {target_nl_name}. '
        
        objects = self.env.last_event.metadata['objects']
        broken_object_list = []
        for o in objects:
            oid = o['objectId']
            oname= o['name']
            obj_class_name = oid.split("|")[0]
            # if obj_class_name in ignore_classes:
            #     continue
            
            if 'Cracked' in oid or 'Cracked' in oname:
                broken_object_list.append(o)
        
        self.update_name_dict_broken(broken_object_list)
        self.update_camera()
        return ret, action_success

    def get_agent_cam_frames(self, option="rgb"):
        """
        Get the agent viewpoint frame of the current scene.
        """
        if option=="rgb":
            return cv2.cvtColor(self.env.last_event.cv2img, cv2.COLOR_BGR2RGB)
        elif option=="depth":
            return cv2.cvtColor(self.env.last_event.depth_frame, cv2.COLOR_BGR2RGB)
        elif option=="semantic":
            return self.env.last_event.semantic_segmentation_frame
        elif option=="instance":
            return self.env.last_event.instance_segmentation_frame
        elif option=="2d_bbox":
            return self.env.last_event.class_detections2D
        elif option=="3d_bbox":
            return self.env.last_event.objects_by_type
        else:
            raise ValueError(f"Invalid option: {option}")
    
    def get_third_party_camera_frames(self, instruction=None):
        self.last_agent_image = self.update_camera_viewpoint(option="agent_view")
        self.last_mapview_image = self.update_camera_viewpoint(option="map_view")
        self.last_agent_cam_rgb_image = self.get_agent_cam_frames(option="rgb")

        if self.is_agent_view_cam_rgb_frame_save:
            # save agent view camera frame
            save_root = os.path.join(
                self.cam_frame_save_path,
                # self._task_type,
                # self.task_name.split(".json")[0],
                "agent_view_frame"
            )
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            save_path = os.path.join(
                save_root,
                f"frame_{self.frame_num}.png"
            )

            # if there are multiple camera properties 
            if isinstance(self.last_agent_image, list):
                for i, img in enumerate(self.last_agent_image):
                    property_save_path = os.path.join(
                        save_root, 
                        self.task_name,
                        "property_{}".format(i)
                    )
                    if not os.path.exists(property_save_path):
                        os.makedirs(property_save_path)

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(property_save_path, f"frame_{self.frame_num}.png"), img)
                    if instruction is not None:
                        # write text in image 
                        txt_img = cv2.putText(
                            img, instruction, (0, 15), 1, 1, (255, 255, 255), 2
                        )
                        cv2.imwrite(os.path.join(property_save_path, f"frame_{self.frame_num}_inst.png"), txt_img)
                    
            else:
                img = cv2.cvtColor(self.last_agent_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(save_path, img)
                if instruction is not None:
                    # write text in image 
                    txt_img = cv2.putText(
                        img, instruction, (0, 15), 1, 1, (255, 255, 255), 2
                    )
                    cv2.imwrite(save_path.replace(".png", "_txt.png"), txt_img)

        if self.is_topview_cam_rgb_frame_save:
            # save topview cam 
            save_root = os.path.join(
                self.cam_frame_save_path,
                # self._task_type,
                # self.task_name.split(".json")[0],
                "top_view_frame"
            )
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            save_path = os.path.join(
                save_root,
                f"frame_{self.frame_num}.png"
            )
            if isinstance(self.last_mapview_image, list):
                img = self.last_mapview_image[-1]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(save_path, img)
                if instruction is not None:
                    # write text in image 
                    txt_img = cv2.putText(
                        img, instruction, (0, 15), 1, 1, (255, 255, 255), 2
                    )
                    cv2.imwrite(save_path.replace(".png", "_txt.png"), txt_img)
            else:
                img = cv2.cvtColor(self.last_mapview_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(save_path, img)
                if instruction is not None:
                    # write text in image 
                    txt_img = cv2.putText(
                        img, instruction, (0, 15), 1, 1, (255, 255, 255), 2
                    )
                    cv2.imwrite(save_path.replace(".png", "_txt.png"), txt_img)
        if self.is_agent_cam_rgb_frame_save:
            # save agent cam rgb frame
            save_root = os.path.join(
                self.cam_frame_save_path,
                # self._task_type,
                # self.task_name.split(".json")[0],
                "agent_cam_rgb_frame"
            )
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            save_path = os.path.join(
                save_root,
                f"frame_{self.frame_num}.png"
            )
            if isinstance(self.last_agent_cam_rgb_image, list):
                img = self.last_agent_cam_rgb_image[-1]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(save_path, img)
                if instruction is not None:
                    # write text in image 
                    txt_img = cv2.putText(
                        img, instruction, (0, 15), 1, 1, (255, 255, 255), 2
                    )
                    cv2.imwrite(save_path.replace(".png", "_txt.png"), txt_img) 
            else:
                img = cv2.cvtColor(self.last_agent_cam_rgb_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(save_path, img)
                if instruction is not None:
                    # write text in image 
                    txt_img = cv2.putText(
                        img, instruction, (0, 15), 1, 1, (255, 255, 255), 2
                    )
                    cv2.imwrite(save_path.replace(".png", "_txt.png"), txt_img)
        self.frame_num +=1
    
    def get_object_prop(self, obj_id, prop_name, metadata):
        for obj in metadata['objects']:
            if obj['objectId'] == obj_id:
                return obj[prop_name]
        return None
    
    def find_close_reachable_position(self, loc, nth=1):
        d, i = self.reachable_position_kdtree.query(loc, k=nth + 1)
        selected = i[nth - 1]
        return self.reachable_positions[selected]

    def get_obj_id_from_name(self, obj_name, obj_num=None, parent_receptacle_penalty=True, priority_in_visibility=False, exclude_obj_id=None, get_inherited=False):
        obj_id = None
        obj_data = None
        min_distance = 1e+8

        if obj_num != None:
            if obj_num < 1:
                log.warning(f'obj_num should be greater than 0')
                return None, None

            if obj_name in self.multi_objs_dict.keys():
                for tmp_id in self.multi_objs_dict[obj_name].keys():
                    tmp_num = self.multi_objs_dict[obj_name][tmp_id]
                    if tmp_num == obj_num:
                        obj_id = tmp_id
                        break

                if obj_id is not None:

                    for obj in self.env.last_event.metadata['objects']:
                        if obj['objectId'] == obj_id:
                            obj_data = obj
                            break
                    return obj_id, obj_data
            
        for obj in self.env.last_event.metadata['objects']:
            if obj['objectId'] == exclude_obj_id:
                continue
            if obj_name in self.multi_objs_dict.keys() and obj['objectId'] in self.multi_objs_dict[obj_name].keys():
                # import pdb; pdb.set_trace()
                continue

            if obj['objectId'].split('|')[0].casefold() == obj_name.casefold() and (get_inherited is False or len(obj['objectId'].split('|')) == 5):

                flag = False
                if obj["distance"] < min_distance:
                    penalty_advantage = 0  # low priority for objects in closable receptacles such as fridge, microwave
                    
                    if parent_receptacle_penalty and obj['parentReceptacles']:
                        for p in obj['parentReceptacles']:
                            is_open = self.get_object_prop(p, 'isOpen', self.env.last_event.metadata)
                            openable = self.get_object_prop(p, 'openable', self.env.last_event.metadata)
                            if openable is True and is_open is False:
                                flag = True
                                break
                    
                    # do not choose objects in close receptacles
                    if flag:
                        continue

                    if obj_name.casefold() == 'stoveburner' or obj_name.casefold() == 'toaster':
                        # try to find an empty stove
                        if len(obj['receptacleObjectIds']) > 0:
                            penalty_advantage += 10000

                    if priority_in_visibility and obj['visible'] is False:
                        penalty_advantage += 1000

                    if obj["distance"] + penalty_advantage < min_distance:
                        min_distance = obj["distance"] + penalty_advantage
                        obj_data = obj
                        obj_id = obj["objectId"]

        if obj_id is not None and obj_num != None:
            # import pdb; pdb.set_trace()
            if obj_name not in self.multi_objs_dict.keys():
                self.multi_objs_dict[obj_name] = {}
            self.multi_objs_dict[obj_name][obj_id] = obj_num

        return obj_id, obj_data
    
    @staticmethod
    def angle_diff(x, y):
        x = math.radians(x)
        y = math.radians(y)
        return math.degrees(math.atan2(math.sin(x - y), math.cos(x - y)))
    
    @staticmethod
    def point_to_array(point):
        """
        AI2THOR point to numpy array.

        Args:
            point (dict or np.ndarray): Point with 'x', 'y', and 'z' or numpy array.

        Returns:
            np.ndarray: Numpy array with x and z coordinates
        """
        if isinstance(point, dict):
            return np.array([point['x'], point['z']])  # y축 제외
        elif isinstance(point, np.ndarray):
            return point[[0, 2]]  # numpy 배열에서 x, z 추출
        return np.array(point[:2])
    
    def find_closest_edge_and_point(self, polygon, point):
        """
        Find the closest edge and point on a polygon to a given point.

        Args:
            polygon (list): List of points defining the polygon.
            point (dict): The point to find the closest edge and point to, with 'x', 'y', and 'z'.

        Returns:
            tuple: Closest edge defined by two points and the closest point on the edge.
        """
        min_distance = float('inf')
        closest_point = None
        closest_edge = None
        num_points = len(polygon)
        for i in range(num_points):
            p1 = self.point_to_array(polygon[i])
            p2 = self.point_to_array(polygon[(i + 1) % num_points])
            proj_point = self.closest_point_on_segment(p1, p2, point)
            dist = np.linalg.norm(proj_point - point)
            if dist < min_distance:
                min_distance = dist
                closest_point = proj_point
                closest_edge = (p1, p2)
        return closest_edge, closest_point
    
    @staticmethod
    def closest_point_on_segment(p1, p2, point):
        """
        Get the closest point on a line segment to a given point.

        Args:
            p1 (np.ndarray): First point of the line segment.
            p2 (np.ndarray): Second point of the line segment.
            point (np.ndarray): Point to find the closest point to.

        Returns:
            np.ndarray: Closest point on the line segment.
        """
        line_vec = p2 - p1
        point_vec = point - p1
        line_len = np.dot(line_vec, line_vec)
        if line_len == 0:
            return p1
        t = np.dot(point_vec, line_vec) / line_len
        t = max(0, min(1, t))
        return p1 + t * line_vec

    def find_closest_positions_to_obj_wall_orthogonal_line(self, floor_polygon, object_position, reachable_positions):
        """
        Find closest positions to face the line orthogonal to the closest wall and passing through the object. 
        Especially used for special objects that need to be placed against the wall
        In this case, when the object is opened, we need to face the object due to check the inside of the object.

        Args:
            floor_polygon (list): List of points defining the floor polygon.
            object_position (dict): Object position with 'x', 'y', and 'z'.
            reachable_positions (list): List of reachable positions to check.

        Returns:
            list: List of closest positions to face the orthogonal line.
        """
        obj_point = self.point_to_array(object_position)

        # 가장 가까운 선분(edge)과 직교 교점 찾기
        closest_edge, orthogonal_intersection = self.find_closest_edge_and_point(floor_polygon, obj_point)
        edge_direction = closest_edge[1] - closest_edge[0]
        edge_direction /= np.linalg.norm(edge_direction)
        orthogonal_direction = np.array([-edge_direction[1], edge_direction[0]])

        # 가장 가까운 reachable position 최대 5개 선택
        closest_positions = self.find_closest_positions_to_obj_wall_line(
            orthogonal_intersection, 
            orthogonal_direction, 
            reachable_positions,
            20
        )
        return closest_positions

    def go_to_obj(self, target_nl_name):
        """
        find reachable position and goto to the target object by nl name
        only called by nav_obj function 

        Args:
            target_nl_name (str): Target object name in natural language.
        
        Returns:
            str: Navigation result message.
        """
        if not target_nl_name in self.high_name_to_obj_id_dict.keys():
            ret_msg = f'Cannot find {target_nl_name}. Is it exist in this house? '
            return ret_msg, False
        
        target_obj_id = self.high_name_to_obj_id_dict[target_nl_name]
        target_obj_class = target_nl_name.split(' ')[0]
        
        if target_obj_id not in self.found_objects:
            ret_msg = f'You have not found {target_nl_name} yet. You need to observe it first. '
            return ret_msg, False
        
        objects = self.env.last_event.metadata['objects']
        obj_pos = None
        obj_data = None
        for o in objects:
            if o['objectId'] == target_obj_id:
                obj_data = o
                break

        if (obj_data is not None) and ('position' in obj_data.keys()):
            obj_pos = obj_data['position']
            obj_rot = obj_data['rotation']['y']
            obj_is_visible = obj_data['visible']
            obj_distance = obj_data['distance']
      
            # do not move if the object is already visible and close
            # if obj_is_visible and obj_distance < 1.0:
            #     return 'Object is already visible and close.'
      
            # normal case 
            ## go to objects 
            # teleport sometimes fails even with reachable positions. 
            # if fails, repeat with the next closest reachable positions.
            max_attempts = 40 #30
            teleport_success = False    
            reachable_pos_idx = 0

            # get obj location

            # do not move if the object is already visible and close
            # if obj_is_visible and obj_distance < 1.0:
            #     log.info('Object is already visible')
            #     max_attempts = 0
            #     teleport_success = True

            # try teleporting
            reachable_pos_idx = 0
            
            for i in range(max_attempts):
                reachable_pos_idx += 1
                if i == 10 and (target_obj_class == 'Fridge' or target_obj_class == 'Microwave'):
                    reachable_pos_idx -= 10

                closest_loc = self.find_close_reachable_position(
                    [obj_pos['x'], obj_pos['y'], obj_pos['z']], 
                    reachable_pos_idx
                )

                # calculate desired rotation angle (see https://github.com/allenai/ai2thor/issues/806)
                rot_angle = math.atan2(-(obj_pos['x'] - closest_loc[0]), obj_pos['z'] - closest_loc[2])
                if rot_angle > 0:
                    rot_angle -= 2 * math.pi
                rot_angle = -(180 / math.pi) * rot_angle  # in degrees

                if i < 10 and (target_obj_class == 'Fridge' or target_obj_class == 'Microwave'):  # not always correct, but better than nothing
                    angle_diff = abs(self.angle_diff(rot_angle, obj_rot))
                    if target_obj_class == 'Fridge' and \
                            not ((90 - 20 < angle_diff < 90 + 20) or (270 - 20 < angle_diff < 270 + 20)):
                        continue
                    if target_obj_class == 'Microwave' and \
                            not ((180 - 20 < angle_diff < 180 + 20) or (0 - 20 < angle_diff < 0 + 20)):
                        continue

                # calculate desired horizon angle
                camera_height = self.agent_height + self.CAMERA_HEIGHT_OFFSET
                xz_dist = math.hypot(obj_pos['x'] - closest_loc[0], obj_pos['z'] - closest_loc[2])
                hor_angle = math.atan2((obj_pos['y'] - camera_height), xz_dist)
                hor_angle = (180 / math.pi) * hor_angle  # in degrees
                hor_angle *= 0.9  # adjust angle for better view
                # hor_angle = -30
                # hor_angle = 0

                # teleport
                self.env.step(
                    "TeleportFull",
                    x=closest_loc[0], 
                    y=self.agent_height, 
                    z=closest_loc[2],
                    rotation=rot_angle, 
                    horizon=-hor_angle, 
                    standing=True,
                    forceAction=True
                )
                # self.env.step(
                #     action="Teleport",
                #     position = {'x': closest_loc[0], 'y': self.agent_height, 'z': closest_loc[2]},
                #     rotation = rot_angle,
                #     horizon = -hor_angle, #30,
                #     standing=True
                # )

                if not self.env.last_event.metadata['lastActionSuccess']:
                    if i == max_attempts - 1:
                        log.warning(f"TeleportFull action failed: {self.env.last_event.metadata['errorMessage']}")
                        break
                else:
                    teleport_success = True
                    break

        if not teleport_success:
            ret_msg = f'Cannot move to {target_nl_name}. '
        else:
            ret_msg = f'You arrive at {target_nl_name}. '

        self.update_camera()

        return ret_msg, teleport_success

    def find(self, target_nl_name):
        """
        find reachable position and goto to the target object by nl name
        only called by nav_obj function 

        Args:
            target_nl_name (str): Target object name in natural language.
        
        Returns:
            str: Navigation result message.
        """
        if not target_nl_name in self.high_name_to_obj_id_dict.keys():
            ret_msg = f'Cannot find {target_nl_name}. Is it exist in this house? '
            return ret_msg, False
        
        target_obj_id = self.high_name_to_obj_id_dict[target_nl_name]
        target_obj_class = target_nl_name.split(' ')[0]
        
        objects = self.env.last_event.metadata['objects']
        obj_pos = None
        obj_data = None
        for o in objects:
            if o['objectId'] == target_obj_id:
                obj_data = o
                break

        if (obj_data is not None) and ('position' in obj_data.keys()):
            obj_pos = obj_data['position']
            obj_rot = obj_data['rotation']['y']
            obj_is_visible = obj_data['visible']
            obj_distance = obj_data['distance']
      
            # do not move if the object is already visible and close
            # if obj_is_visible and obj_distance < 1.0:
            #     return 'Object is already visible and close.'
      
            # normal case 
            ## go to objects 
            # teleport sometimes fails even with reachable positions. 
            # if fails, repeat with the next closest reachable positions.
            max_attempts = 40 #30
            teleport_success = False    
            reachable_pos_idx = 0

            # get obj location

            # do not move if the object is already visible and close
            # if obj_is_visible and obj_distance < 1.0:
            #     log.info('Object is already visible')
            #     max_attempts = 0
            #     teleport_success = True

            # try teleporting
            reachable_pos_idx = 0
            
            for i in range(max_attempts):
                reachable_pos_idx += 1
                if i == 10 and (target_obj_class == 'Fridge' or target_obj_class == 'Microwave'):
                    reachable_pos_idx -= 10

                closest_loc = self.find_close_reachable_position(
                    [obj_pos['x'], obj_pos['y'], obj_pos['z']], 
                    reachable_pos_idx
                )

                # calculate desired rotation angle (see https://github.com/allenai/ai2thor/issues/806)
                rot_angle = math.atan2(-(obj_pos['x'] - closest_loc[0]), obj_pos['z'] - closest_loc[2])
                if rot_angle > 0:
                    rot_angle -= 2 * math.pi
                rot_angle = -(180 / math.pi) * rot_angle  # in degrees

                if i < 10 and (target_obj_class == 'Fridge' or target_obj_class == 'Microwave'):  # not always correct, but better than nothing
                    angle_diff = abs(self.angle_diff(rot_angle, obj_rot))
                    if target_obj_class == 'Fridge' and \
                            not ((90 - 20 < angle_diff < 90 + 20) or (270 - 20 < angle_diff < 270 + 20)):
                        continue
                    if target_obj_class == 'Microwave' and \
                            not ((180 - 20 < angle_diff < 180 + 20) or (0 - 20 < angle_diff < 0 + 20)):
                        continue

                # calculate desired horizon angle
                camera_height = self.agent_height + self.CAMERA_HEIGHT_OFFSET
                xz_dist = math.hypot(obj_pos['x'] - closest_loc[0], obj_pos['z'] - closest_loc[2])
                hor_angle = math.atan2((obj_pos['y'] - camera_height), xz_dist)
                hor_angle = (180 / math.pi) * hor_angle  # in degrees
                hor_angle *= 0.9  # adjust angle for better view
                # hor_angle = -30
                # hor_angle = 0

                # teleport
                self.env.step(
                    "TeleportFull",
                    x=closest_loc[0], 
                    y=self.agent_height, 
                    z=closest_loc[2],
                    rotation=rot_angle, 
                    horizon=-hor_angle, 
                    standing=True,
                    forceAction=True
                )
                # self.env.step(
                #     action="Teleport",
                #     position = {'x': closest_loc[0], 'y': self.agent_height, 'z': closest_loc[2]},
                #     rotation = rot_angle,
                #     horizon = -hor_angle, #30,
                #     standing=True
                # )

                if not self.env.last_event.metadata['lastActionSuccess']:
                    if i == max_attempts - 1:
                        log.warning(f"TeleportFull action failed: {self.env.last_event.metadata['errorMessage']}")
                        break
                else:
                    teleport_success = True
                    break

        if not teleport_success:
            ret_msg = f'Cannot move to {target_nl_name}. '
        else:
            ret_msg = f'You arrive at {target_nl_name}. '

        self.update_camera()

        return ret_msg, teleport_success

    def break_(self, obj_name, obj_num):
        log.info(f'break {obj_name}')
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name, obj_num=obj_num)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to break'
        else:
            self.env.step(
                action="BreakObject",
                objectId=obj_id,
            )

            if not self.env.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Break action failed"

        return ret_msg
    
    def drop(self):
        """
        Drop action wrapper function

        Args:
            target_nl_name (str): Target object name in natural language.

        Returns:
            str: Drop result message.
        """
        ret_msg = ''
        action_success=False
        if len(self.env.last_event.metadata['inventoryObjects']) == 0:
            return 'Nothing Done. Robot is not holding any object. ', action_success
        self.env.step(action="UnpausePhysicsAutoSim")
        for j in range(16):
            # looks silly but it works
            if j == 1:
                self.env.step(dict(action="LookUp", degrees=15))
            elif j == 2:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 3:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 4:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 5:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 6:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 7:
                self.env.step(dict(action="LookDown"), degrees=55)
            elif j == 8:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 9:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 10:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 11:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 12:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 13:
                self.env.step(dict(action="LookUp"), degrees=40)
                self.env.step(dict(action="LookUp"))
                self.env.step(dict(action="LookUp"))
                self.env.step(dict(action="MoveBack"))
            elif j == 14:
                self.env.step(dict(action="MoveAhead"))
                for r in range(4):
                    self.env.step(dict(action="MoveRight"))
            elif j == 15:
                for r in range(8):
                    self.env.step(dict(action="MoveLeft"))
            elif j == 16:
                for r in range(4):
                    self.env.step(dict(action="MoveRight"))
                self.env.step(dict(  # this somehow make putobject success in some cases
                    action="RotateRight",
                    degrees=15
                ))

            self.env.step(
                action="DropHandObject",
                forceAction=True
            )
            if not self.env.last_event.metadata['lastActionSuccess']:
                if j == 16:
                    log.warning(f"Drop action failed, the error message is {self.env.last_event.metadata['errorMessage']}")
                    ret_msg = f"Drop action failed. "
            else:
                ret_msg = 'You drop the object. '
                action_success=True
                break
        self.env.step(action="PausePhysicsAutoSim")
        self.update_camera()
        return ret_msg, action_success
    
    def throw(self):
        """
        Throw action wrapper function

        Args:
            target_nl_name (str): Target object name in natural language.

        Returns:
            str: Throw result message.
        """
        ret_msg = ''
        action_success=False
        if len(self.env.last_event.metadata['inventoryObjects']) == 0:
            return 'Nothing Done. Robot is not holding any object. '

        for j in range(16):
            # looks silly but it works
            if j == 1:
                self.env.step(dict(action="LookUp", degrees=15))
            elif j == 2:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 3:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 4:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 5:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 6:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 7:
                self.env.step(dict(action="LookDown"), degrees=55)
            elif j == 8:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 9:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 10:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 11:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 12:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 13:
                self.env.step(dict(action="LookUp"), degrees=40)
                self.env.step(dict(action="LookUp"))
                self.env.step(dict(action="LookUp"))
                self.env.step(dict(action="MoveBack"))
            elif j == 14:
                self.env.step(dict(action="MoveAhead"))
                for r in range(4):
                    self.env.step(dict(action="MoveRight"))
            elif j == 15:
                for r in range(8):
                    self.env.step(dict(action="MoveLeft"))
            elif j == 16:
                for r in range(4):
                    self.env.step(dict(action="MoveRight"))
                self.env.step(dict(  # this somehow make putobject success in some cases
                    action="RotateRight",
                    degrees=15
                ))
            
            self.env.step(
                action="ThrowObject",
                moveMagnitude=1500.0,
                forceAction=True
            )
            if not self.env.last_event.metadata['lastActionSuccess']:
                if j == 16:
                    log.warning(f"Throw action failed, the error message is {self.env.last_event.metadata['errorMessage']}")
                    ret_msg = f"Throw action failed. "
            else:
                action_success=True
                ret_msg = 'You throw the object. '
                break

        return ret_msg, action_success

    def _turn_to_obj(self, target_obj_pos):
        """
        turning agent to the target position 
        """
        # turn to the target object
        max_attempts = 20
        reachable_pos_idx = 0
        for i in range(max_attempts):
            reachable_pos_idx += 1
            closest_loc = self.find_close_reachable_position(
                [target_obj_pos['x'], target_obj_pos['y'], target_obj_pos['z']], 
                reachable_pos_idx
            )
            # calculate desired rotation angle 
            # (see https://github.com/allenai/ai2thor/issues/806)
            rot_angle = math.atan2(
                -(target_obj_pos['x'] - closest_loc[0]), 
                target_obj_pos['z'] - closest_loc[2]
            )
            if rot_angle > 0:
                rot_angle -= 2 * math.pi
            rot_angle = -(180 / math.pi) * rot_angle  # in degrees
            # turn to obj 
            self.env.step(
                action="RotateRight",
                degrees=rot_angle
            )
            if self.env.last_event.metadata['lastActionSuccess']:
                break

    def pour(self, target_nl_name):
        """
        Pouring action wrapper function 

        Args:
            target_nl_name (str): Target object name in natural language.

        Returns:
            str: Pour result message.
        """
        ret_msg = ''
        action_success=False
        target_nl_name = target_nl_name.strip()

        if len(self.env.last_event.metadata['inventoryObjects']) == 0:
            return 'Nothing Done. Robot is not holding any object. ', action_success
        
        # move item right few steps to mitigate occlusion issue when pouring 
        self.env.step(
            "MoveHeldObjectRight",
            moveMagnitude=0.3,
            forceVisible=False
        )

        # check visibility 
        visible_objects = self.get_visible_obj_nl_names_from_last_event()
        if target_nl_name not in visible_objects:
            return f'Nothing Done. {target_nl_name} is not visible. ', action_success

        obj_id = self.env.last_event.metadata['inventoryObjects'][0]['objectId']
        obj_inf = self.get_obj_information(obj_id) 
        source_liquid_type = obj_inf['fillLiquid']
        if source_liquid_type is None:
            obj_class = obj_id.split('|')[0]
            is_filled = obj_inf['isFilledWithLiquid']
            if is_filled:
                if "Wine" in obj_class:
                    source_liquid_type = "wine"

        # fillLiquid = liquidtype 

        # print(obj_inf.keys())
        # print('canFillWithLiquid', obj_inf['canFillWithLiquid'])
        # print('isFilledWithLiquid', obj_inf['isFilledWithLiquid'])
        # print('fillLiquid', obj_inf['fillLiquid'])
        # print('temperature', obj_inf['temperature'])
        # print('objectType', obj_inf['objectType'])
        # import sys
        # sys.exit()

        if obj_inf is None:
            return 'Nothing Done. Cannot find the object. ', action_success

        can_pour = obj_inf['canFillWithLiquid']
        if not can_pour:
            return 'Nothing Done. You can not pour the object you are holding. ', action_success
        
        is_filled = obj_inf['isFilledWithLiquid']
        if not is_filled:
            # object is not filled with liquid --> do nothing
            return 'Nothing Done. The object you holding is not filled with liquid. ', action_success
        else:
            # Do pouring motion : Don't care about the target object
            # just animate the pouring motion
            #angel_list = [60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0]

            self.env.step(action="UnpausePhysicsAutoSim")
            # turn to the target object
            if not target_nl_name in self.high_name_to_obj_id_dict.keys():
                for k in self.high_name_to_obj_id_dict.keys():
                    print(k)
                print("can not find target object ", target_nl_name)
            target_obj_id = self.high_name_to_obj_id_dict[target_nl_name]
            target_obj_inf = self.get_obj_information(target_obj_id)
            target_obj_pos = target_obj_inf['position']
            target_obj_class = target_obj_id.split('|')[0]

            if target_obj_inf['canFillWithLiquid'] is True:

                # if not target_obj_class in utils.STATIC_RECEPTACLES:
                #     self._turn_to_obj(target_obj_pos)

                angel_list = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 30.0, -30.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]

                for angel in angel_list:
                    self.env.step(
                        action="RotateHeldObject",
                        pitch=angel
                    )
                    self.get_third_party_camera_frames()

                # Change the object states 
                # 1. empty the holding object 
                self.env.step(
                    action="EmptyLiquidFromObject",
                    objectId=obj_id
                )
                # TODO: change to the actual liquid type by checking the object
                # wine, water, coffee

                # 2. Fill the target object with liquid
                target_obj_id = self.high_name_to_obj_id_dict[target_nl_name]
                holding_obj_id = self.env.last_event.metadata['inventoryObjects'][0]['objectId']

                # get target_obj_position 
                target_obj_inf = self.get_obj_information(target_obj_id)
                target_obj_pos = target_obj_inf['position']

                if target_obj_inf['canFillWithLiquid'] is True:
                    self.env.step(
                        action="FillObjectWithLiquid",
                        objectId=target_obj_id,
                        fillLiquid=source_liquid_type, 
                    )
                    if not self.env.last_event.metadata['lastActionSuccess']:
                        ret_msg = f'Pouring failed. '
                    else:
                        action_success=True
                        ret_msg = f'Pouring success. '
                    
                    if source_liquid_type == "coffee" or source_liquid_type =="wine":
                        # make the object dirty 
                        self.env.step(
                            action="DirtyObject",
                            objectId=holding_obj_id
                        )
            else:
                # # just done action 
                # if target_obj_class in utils.STATIC_RECEPTACLES:
                #     self.env.step(
                #         action="DirtyObject",
                #         objectId=target_obj_id,
                #     )
                #     if not self.env.last_event.metadata['lastActionSuccess']:
                #         ret_msg = f'Pouring failed. '
                #     else:
                #         action_success=True
                #         ret_msg = f'Pouring success. '
                # else:
                # if not self.env.last_event.metadata['lastActionSuccess']:
                #     ret_msg = f'Pouring failed. '
                # else:
                #     action_success=True
                #     ret_msg = f'Pouring success. '
                ret_msg = f'You can not pouring liquid into {target_nl_name}. Pouring failed. '
            
        self.env.step(action="PausePhysicsAutoSim")

        self.update_camera()
        return ret_msg, action_success
    
    def unchanged(self):
        self.env.step(
                action="RotateHeldObject",
                pitch=0.0
        )
        return
    
    def done(self):
        self.env.step(
                action="Done",
        )
        return

    def put(self, receptacle_name, obj_num):
        # assume the agent always put the object currently holding
        ret_msg = ''

        if len(self.env.last_event.metadata['inventoryObjects']) == 0:
            ret_msg = f'Nothing Done. Robot is not holding any object'
            return ret_msg
        else:
            holding_obj_id = self.env.last_event.metadata['inventoryObjects'][0]['objectId']

        halt = False
        last_recep_id = None
        exclude_obj_id = None
        for k in range(2):  # try closest and next closest one
            for j in range(17):  # move/look around or rotate obj
                for i in range(2):  # try inherited receptacles too (e.g., sink basin, bath basin)
                    if k == 1 and exclude_obj_id is None:
                        exclude_obj_id = last_recep_id  # previous recep id

                    if i == 0:
                        recep_id, _ = self.get_obj_id_from_name(receptacle_name, exclude_obj_id=exclude_obj_id, obj_num=obj_num)
                    else:
                        recep_id, _ = self.get_obj_id_from_name(receptacle_name, get_inherited=True, exclude_obj_id=exclude_obj_id, obj_num=obj_num)

                    if not recep_id:
                        ret_msg = f'Cannot find {receptacle_name} {obj_num}'
                        continue

                    log.info(f'put {holding_obj_id} on {recep_id}')
                    flag = False
                    # look up (put action fails when a receptacle is not visible)

                    if j == 1:
                        self.env.step(dict(action="LookUp", degrees=15))
                    elif j == 2:
                        self.env.step(dict(action="LookUp"), degrees=5)
                    elif j == 3:
                        self.env.step(dict(action="LookUp"), degrees=5)
                    elif j == 4:
                        self.env.step(dict(action="LookUp"), degrees=5)
                    elif j == 5:
                        self.env.step(dict(action="LookUp"), degrees=5)
                    elif j == 6:
                        self.env.step(dict(action="LookUp"), degrees=5)
                    elif j == 7:
                        self.env.step(dict(action="LookDown"), degrees=55)
                    elif j == 8:
                        self.env.step(dict(action="LookDown"), degrees=5)
                    elif j == 9:
                        self.env.step(dict(action="LookDown"), degrees=5)
                    elif j == 10:
                        self.env.step(dict(action="LookDown"), degrees=5)
                    elif j == 11:
                        self.env.step(dict(action="LookDown"), degrees=5)
                    elif j == 12:
                        self.env.step(dict(action="LookDown"), degrees=5)
                    elif j == 13:
                        self.env.step(dict(action="LookUp"), degrees=40)
                        self.env.step(dict(action="LookUp"))
                        self.env.step(dict(action="LookUp"))
                        self.env.step(dict(action="MoveBack"))
                    elif j == 14:
                        self.env.step(dict(action="MoveAhead"))
                        for r in range(4):
                            self.env.step(dict(action="MoveRight"))
                    elif j == 15:
                        for r in range(8):
                            self.env.step(dict(action="MoveLeft"))
                    elif j == 16:
                        for r in range(4):
                            self.env.step(dict(action="MoveRight"))
                        self.env.step(dict(  # this somehow make putobject success in some cases
                            action="RotateRight",
                            degrees=15
                        ))
                    elif j == 17:
                        event = self.env.step(
                            action="GetSpawnCoordinatesAboveReceptacle",
                            objectId=recep_id,
                            anywhere=False
                        )
                        position_above = event.metadata['actionReturn']
                        self.env.step(
                            action="PlaceObjectAtPoint",
                            objectId=holding_obj_id,
                            position = {
                                "x": sum([tmp['x'] for tmp in position_above])/len(position_above),
                                "y": sum([tmp['y'] for tmp in position_above])/len(position_above),
                                "z": sum([tmp['z'] for tmp in position_above])/len(position_above)
                            }
                        )
                        obj_info = self.get_obj_information(holding_obj_id)
                        flag = True

                    last_recep_id = recep_id
                    if not flag:
                        self.env.step(dict(
                            action="PutObject",
                            objectId=recep_id,
                            forceAction=True
                        ))
                    
                        if not self.env.last_event.metadata['lastActionSuccess']:
                            if j == 16:
                                log.warning(f"PutObject action failed: {self.env.last_event.metadata['errorMessage']}, trying again...")
                                ret_msg = f'Putting the object on {receptacle_name} failed. '
                        else:
                            ret_msg = ''
                            halt = True
                            break
                    else:

                        if recep_id in obj_info['parentReceptacles']:
                            ret_msg = ''
                            halt = True
                            break
                        else:
                            log.warning(f"PutObject action failed: {self.env.last_event.metadata['errorMessage']}, trying again...")
                            ret_msg = f'Putting the object on {receptacle_name} failed. '
                            
                if halt:
                    break
            if halt:
                break

        return ret_msg

    def gen_sub_name_dict_from_obj_list(self, obj_list):
        ignore_classes=["Floor"]
        obj_id_to_high_name_dict = {}
        obj_name_to_high_name_dict = {}
        high_name_to_obj_id_dict = {}
        high_name_to_obj_name_dict = {}
        
        name_counter = {}  # 클래스별 이름 카운트

        # 객체 처리
        for obj in obj_list:
            obj_id = obj['objectId']
            obj_name = obj['name']
            obj_class_name = obj_id.split('|')[0]
            if obj_class_name in ignore_classes:
                continue
            
            # class counter 
            if obj_class_name not in name_counter.keys():
                name_counter[obj_class_name] = 0
            
            name_counter[obj_class_name] += 1
            high_name = obj_class_name + " ({})".format(name_counter[obj_class_name])

            obj_id_to_high_name_dict[obj_id] = high_name
            obj_name_to_high_name_dict[obj_name] = high_name
            high_name_to_obj_id_dict[high_name] = obj_id
            high_name_to_obj_name_dict[high_name] = obj_name
        
        return obj_id_to_high_name_dict, obj_name_to_high_name_dict, high_name_to_obj_id_dict, high_name_to_obj_name_dict


    def slice(self, target_nl_name):
        """
        Slice action wrapper function

        Args:
            target_nl_name (str): Target object name in natural language.

        Returns:
            str: Slice result message.
        """
        log.info(f'slice {target_nl_name}')
        ret_msg = ''
        action_success=False

        if target_nl_name not in self.high_name_to_obj_id_dict.keys():
            ret_msg = f'Cannot find {target_nl_name} to slice. '
            return ret_msg, action_success
        
        obj_id = self.high_name_to_obj_id_dict[target_nl_name]
        
        # check holding knife 
        if len(self.env.last_event.metadata['inventoryObjects']) == 0:
            ret_msg = f'You are not holding a tool that can be used for cutting. '
            return ret_msg, action_success

        # [{'objectId': 'Spoon|surface|6|33', 'objectType': 'Spoon'}]
        holding_obj_id = self.env.last_event.metadata['inventoryObjects'][0]['objectId']
        holding_obj_type = self.env.last_event.metadata['inventoryObjects'][0]['objectType']
        
        # must be hold knife or any cutting tool
        if holding_obj_type not in utils.AI2THOR_CUTTING_TOOL:
            ret_msg = f'You are not holding a tool that can be used for cutting. '
            return ret_msg, action_success

        objects = self.env.last_event.metadata['objects']
        for o in objects:
            if o['objectId'] == obj_id:
                obj_data = o
                break

        # slice the object
        self.env.step(
            action="SliceObject",
            objectId=obj_id,
        )
        # self.update_camera()

        if not self.env.last_event.metadata['lastActionSuccess']:
            if obj_data['isSliced']:
                ret_msg = f'The {target_nl_name} already sliced. '
            elif not obj_data['sliceable']:
                ret_msg = f'The {target_nl_name} cannot be sliced. '
            elif not obj_data['visible']:
                ret_msg = f'The {target_nl_name} is not close to you. '
            else:
                log.info(self.env.last_event.metadata['errorMessage'])
                ret_msg = f'The {target_nl_name} slice failed. '
        else:
            # update the sliced object information
            slice_obj_list = []                
            slice_obj_id_list = []
            for o in self.env.last_event.metadata['objects']:
                # last_name = o['name'].split("|")[-1]
                if '_Slice_' in o['name'] or '_Sliced' in o['name'] or 'Sliced' in o['name'] or 'Slice' in o['name']: 
                    slice_obj_list.append(o['name'])
                    slice_obj_id_list.append(o['objectId'])

            # update all_obj_nlname_to_obj_id_dict and all_obj_obj_id_to_nlname_dict
            # due to the new sliced objects !! 
            self.update_name_dict_slice(slice_obj_id_list)
            
            ret_msg = f'You slice {target_nl_name}. ' 
            action_success=True

        self.update_camera()
        return ret_msg, action_success   

    def update_name_dict_slice(self, slice_obj_list):
        """
        Update self.name_dict
        Arge:
            object_id_list (list): List of object IDs

        Returns:
            None
        """
        # 'Potato_11_Slice_1', 'Potato_11_Slice_2', 'Potato_11_Slice_3', 
        # 'Potato_11_Slice_4', 'Potato_11_Slice_5', 'Potato_11_Slice_6', '
        # Potato_11_Slice_7', 'Potato_11_Slice_8', 'Potato_11_Slice_9
        obj_list = []
        for o in self.env.last_event.metadata['objects']: 
            if o['objectId'] in slice_obj_list:
                obj_list.append(o)
        
        for o in obj_list:
            obj_id = o['objectId']
            obj_name = o['name']
            obj_class_name = obj_id.split('|')[0]+"Sliced"

            if obj_id not in self.obj_id_to_high_name_dict.keys():
                # class counter 
                if obj_class_name not in self.name_counter.keys():
                    self.name_counter[obj_class_name] = 0
                
                self.name_counter[obj_class_name] += 1
                high_name = obj_class_name + " ({})".format(self.name_counter[obj_class_name])

                self.obj_id_to_high_name_dict[obj_id] = high_name
                self.obj_name_to_high_name_dict[obj_name] = high_name
                self.high_name_to_obj_id_dict[high_name] = obj_id
                self.high_name_to_obj_name_dict[high_name] = obj_name

    def update_name_dict_broken(self, broken_obj_list):
        """
        Update self.name_dict
        Arge:
            object_id_list (list): List of object IDs

        Returns:
            None
        """
        
        for o in broken_obj_list:
            obj_id = o['objectId']
            obj_name = o['name']
            obj_class_name = obj_id.split('|')[0]+"Cracked"

            if obj_id not in self.obj_id_to_high_name_dict.keys():
                # class counter 
                if obj_class_name not in self.name_counter.keys():
                    self.name_counter[obj_class_name] = 0
                
                self.name_counter[obj_class_name] += 1
                high_name = obj_class_name + " ({})".format(self.name_counter[obj_class_name])

                self.obj_id_to_high_name_dict[obj_id] = high_name
                self.obj_name_to_high_name_dict[obj_name] = high_name
                self.high_name_to_obj_id_dict[high_name] = obj_id
                self.high_name_to_obj_name_dict[high_name] = obj_name

 
    def cook(self, obj_name, obj_num):
        log.info(f'cook {obj_name}')
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name, obj_num=obj_num)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to cook'
        else:
            self.env.step(
                action="CookObject",
                objectId=obj_id,
            )

            if not self.env.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Cook action failed"

        return ret_msg
    
    def dirty(self, obj_name, obj_num):
        log.info(f'dirty {obj_name}')
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name, obj_num=obj_num)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to dirty'
        else:
            self.env.step(
                action="DirtyObject",
                objectId=obj_id,
            )

            if not self.env.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Dirty action failed"

        return ret_msg
    
    def clean(self, obj_name, obj_num):
        log.info(f'clean {obj_name}')
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name, obj_num=obj_num)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to clean'
        else:
            self.env.step(
                action="CleanObject",
                objectId=obj_id,
            )

            if not self.env.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Clean action failed"

        return ret_msg

    def find_first_child_object_id_list(self, parent_obj_id):
        """
        Find child object IDs from the parent object ID.

        Args:
            parent_obj_id (str): Parent object ID.

        Returns:
            list: List of child object IDs.
        """
        child_obj_id_list = []
        latest_objects = self.env.last_event.metadata['objects']
        for _o in latest_objects:
            parent_receps = _o['parentReceptacles']
            if parent_receps is None:
                continue
            _o_id = _o['objectId']
            _o_class = _o_id.split('|')[0]
            first_parent = parent_receps[-1]
            if first_parent == parent_obj_id:
                child_obj_id_list.append(_o_id)
        
        return child_obj_id_list
    
    def turnon(self, target_nl_name):
        """
        Turn on action wrapper function

        Args:
            target_nl_name (str): Target object name in natural language.

        Returns:
            str: Toggle on result message.
        """
        log.info(f'turn on {target_nl_name}')
        ret_msg = ''
        action_success=False

        if not target_nl_name in self.high_name_to_obj_id_dict.keys():
            ret_msg = f'Cannot find {target_nl_name} to turn on. '
            return ret_msg, action_success

        if len(self.env.last_event.metadata['inventoryObjects']) > 0:
            holding_obj_id = self.env.last_event.metadata['inventoryObjects'][0]['objectId']
            if holding_obj_id is not None:
                log.warning(f'You can not turn on {target_nl_name} while holding an object. Please put it down first.')
                ret_msg = f'You can not turn on {target_nl_name} while holding an object. Please put it down first.'
                return ret_msg, action_success
        
        obj_id = self.high_name_to_obj_id_dict[target_nl_name]
    
        # check it is already turn on or not 
        objects = self.env.last_event.metadata['objects']
        for o in objects:
            if o['objectId'] == obj_id:
                if o['isToggled']:
                    ret_msg = f'{target_nl_name} is already turned on. Nothing done. '
                    action_success = True
                    return ret_msg, action_success
                break

        self.env.step(action="UnpausePhysicsAutoSim")
        self.env.step(
            action="ToggleObjectOn",
            objectId=obj_id,
        )

        if not self.env.last_event.metadata['lastActionSuccess']:
            ret_msg = f"Turn on action failed. "
            action_success=False
        else:
            ret_msg = f'The {target_nl_name} is turned on. '
            action_success=True
        
        if action_success:
            if "Microwave" in target_nl_name:
                child_obj_id_list = self.find_first_child_object_id_list(obj_id)
                for child_obj_id in child_obj_id_list:
                    child_obj_info = self.get_obj_information(child_obj_id)
                    if child_obj_info['cookable']:
                        # cook 
                        self.env.step(
                            action="CookObject",
                            objectId=child_obj_id,
                            forceAction=True
                        )
        
        # target_obj_info = self.get_obj_information(obj_id)

        # if faucet then forcely make object clean 
        if "Faucet" in target_nl_name:
            
            latest_objects = self.env.last_event.metadata['objects']
            sinkbasin_list = []
            for _o in latest_objects:
                _oid = _o['objectId']
                high_name = self.obj_id_to_high_name_dict.get(_oid, None)
                if high_name is None:
                    print(_oid, "not in obj_id_to_high_name_dict")
                    continue
                if "SinkBasin" in high_name:
                    sinkbasin_list.append(_oid)
            
            print("total sink basin count: ", len(sinkbasin_list))
            if len(sinkbasin_list) == 1:
                # find childs 
                childs = []
                for _o in latest_objects:
                    _oid = _o['objectId']
                    _parent = _o['parentReceptacles']
                    if _parent is None:
                        print(_oid, "has no parent receptacles")
                        continue
                    for _p in _parent:
                        if _p == sinkbasin_list[0]:
                            childs.append(_oid)


                print("childs in sink basin: ", childs)
                if len(childs) > 0:
                    for _c in childs:
                        # clean the child object 
                        self.env.step(
                            action="CleanObject",
                            objectId=_c,
                            forceAction=True
                        )
                        if not self.env.last_event.metadata['lastActionSuccess']:
                            log.warning(f"CleanObject action failed: {self.env.last_event.metadata['errorMessage']}")
                        else:
                            print(f"CleanObject action success: {_c} is cleaned.")

            else:
                pass

            # contain_recep_id = None
            # latest_objects = self.env.last_event.metadata['objects']
            # for _o in latest_objects:
            #     if obj_id == _o['objectId']:
            #         parent_receps = _o['parentReceptacles']
            #         if parent_receps is not None:
            #             for recep_id in parent_receps:
            #                 if "Sink" in recep_id:
            #                     contain_recep_id = recep_id
            #                     break
            # # check there are any items in contain_recep 
            # contain_item_ids=[]
            # for _o in latest_objects:
            #     _parent_info = _o['parentReceptacles']
            #     if _parent_info is None:
            #         continue
            #     for p in _parent_info:
            #         if p == contain_recep_id:
            #             contain_item_ids.append(_o['objectId'])
                        
            # for _item_id in contain_item_ids:
            #     if not "Faucet" in _item_id:
            #         self.env.step(
            #             action="CleanObject",
            #             objectId=_item_id,
            #         )

        # NOT WORKING..!! Sink and SinkBasin even can not filled with liquid!!  :( This is AI2THOR Bug 
        # if "Faucet" in target_nl_name:
        #     # 1. find sinkbasin 
        #     sinkbasin_obj_list = []
        #     for _o in latest_objects:
        #         if "SinkBasin" in _o['objectId'].split("|")[-1]:
        #             sinkbasin_obj_list.append(_o['objectId'])
            
        #     if len(sinkbasin_obj_list) == 1:
        #         print(f"Found sink basin: {sinkbasin_obj_list[0]}")
        #         sinkbasin_obj_id = sinkbasin_obj_list[0]
        #         self.env.step(
        #             action="FillObjectWithLiquid",
        #             objectId=sinkbasin_obj_id,
        #             fillLiquid="water",
        #             forceAction=True
        #         )
        #         if not self.env.last_event.metadata['lastActionSuccess']:
        #             print(f"FillObjectWithLiquid action failed: {self.env.last_event.metadata['errorMessage']}")
        #         else:
        #             print(f"FillObjectWithLiquid action success: {sinkbasin_obj_id} is filled with water.")
        #     else:
        #         # TODO: handle multiple sink basins
        #         # maybe using distance? 
        #         # currently, turn on faucet and fill sink basin is not supported in AI2THOR
        #         log.warning(f"Multiple sink basins found: {sinkbasin_obj_list}. Currently not supported.")

        self.update_camera()

        # record last turnon event 
        cooking_appliances = ["Toaster", "StoveKnob", "Microwave"]
        for ca in cooking_appliances:

            if ca in target_nl_name:
               
                turn_on_obj_list = []

                if "Knob" in target_nl_name: # for StoveKnob 
                    control_obj = o['controlledObjects']
                    for e in control_obj: turn_on_obj_list.append(e)
                else:
                    turn_on_obj_list.append(self.high_name_to_obj_id_dict[target_nl_name])

                # 2. get child list 
                target_on_event_data = []
                for on_obj_id in turn_on_obj_list:
                    data_dict = dict()
                    child = []
                    for o in objects:
                        obj_id = o['objectId']
                        obj_class_name = obj_id.split('|')[0]
                        parent_receps = o['parentReceptacles']
                        if not parent_receps is None:
                            for p in parent_receps:
                                if p == on_obj_id:
                                    child.append(obj_id)
                    data_dict['objectId'] = on_obj_id
                    data_dict['child'] = child
                    target_on_event_data.append(data_dict)
                    
                self.last_turnon_event.append(target_on_event_data)
                    

        self.env.step(action="PausePhysicsAutoSim")
        return ret_msg, action_success
    
    def turnoff(self, target_nl_name):
        """
        Turn off action wrapper function

        Args:
            target_nl_name (str): Target object name in natural language.

        Returns:
            str: Toggle off result message.
        """
        log.info(f'turn off {target_nl_name}')
        ret_msg = ''
        action_success=False

        if not target_nl_name in self.high_name_to_obj_id_dict.keys():
            ret_msg = f'Cannot find {target_nl_name} to turn off. '
            return ret_msg, action_success
    
        if len(self.env.last_event.metadata['inventoryObjects']) > 0:
            holding_obj_id = self.env.last_event.metadata['inventoryObjects'][0]['objectId']
            if holding_obj_id is not None:
                log.warning(f'You can not turn off {target_nl_name} while holding an object. Please put it down first.')
                ret_msg = f'You can not turn off {target_nl_name} while holding an object. Please put it down first.'
                return ret_msg, action_success
        
        obj_id = self.high_name_to_obj_id_dict[target_nl_name]
        # check it is already turn off or not 
        objects = self.env.last_event.metadata['objects']
        for o in objects:
            if o['objectId'] == obj_id:
                if not o['isToggled']:
                    ret_msg = f'{target_nl_name} is already turned off. Nothing done. '
                    action_success = True
                    return ret_msg, action_success
                break
        
        self.env.step(action="UnpausePhysicsAutoSim")
        self.env.step(
            action="ToggleObjectOff",
            objectId=obj_id,
        )
        self.env.step(action="PausePhysicsAutoSim")

        if not self.env.last_event.metadata['lastActionSuccess']:
            ret_msg = f"Turn off action failed. "
            
        else:
            ret_msg = f'The {target_nl_name} is turned off. '
            action_success=True

        self.update_camera()
        return ret_msg, action_success
    
    def close(self, target_nl_name):
        """
        Close action wrapper function

        Args:
            target_nl_name (str): Target object name in natural language.
        
        Returns:
            str: Close result message.
        """
        log.info(f'close {target_nl_name}')
        ret_msg = ''
        action_success=False

        if target_nl_name not in self.high_name_to_obj_id_dict.keys():
            ret_msg = f'Cannot find {target_nl_name} to close. '
            return ret_msg, action_success
        
        if len(self.env.last_event.metadata['inventoryObjects']) > 0:
            holding_obj_id = self.env.last_event.metadata['inventoryObjects'][0]['objectId']
            if holding_obj_id is not None:
                log.warning(f'You can not close {target_nl_name} while holding an object. Please put it down first.')
                ret_msg = f'You can not close {target_nl_name} while holding an object. Please put it down first.'
                return ret_msg, action_success
        
        obj_id = self.high_name_to_obj_id_dict[target_nl_name]


        self.env.step(action="UnpausePhysicsAutoSim")
        self.env.step(
            action="CloseObject",
            objectId=obj_id,
        )
        self.env.step(action="PausePhysicsAutoSim")

        # if open action fails, back and try again
        max_move_step = 4 # mostly it works with 1 step
        if not self.env.last_event.metadata['lastActionSuccess']:
            log.warning(
                f"CloseObject action failed: {self.env.last_event.metadata['errorMessage']}, moving backward and trying again...")
            
            ret_msg = f"Close action failed. " 

            for move_step in range(max_move_step):
                # looks silly but it works
                log.warning(
                f"Try to move backward and try again... {move_step+1}/{max_move_step}")

                for _ in range (move_step):
                    self.env.step(action="MoveBack")

                self.env.step(
                    action="CloseObject",
                    objectId=obj_id,
                )

                for _ in range (move_step):
                    self.env.step(action="MoveAhead")
                
                if self.env.last_event.metadata['lastActionSuccess']:
                    log.warning(f"CloseObject action success")
                    ret_msg = ''
                    action_success=True
                    break

            log.warning(f"CloseObject action failed: {self.env.last_event.metadata['errorMessage']}")
            return ret_msg, action_success
        else:
            action_success=True
            ret_msg = f'The {target_nl_name} is closed. '
        
        self.update_camera()
        return ret_msg, action_success

    def open(self, target_nl_name):
        """
        Open action wrapper function

        Args:
            target_nl_name (str): Target object name in natural language.

        Returns:
            str: Open result message.
        """
        log.info(f'open {target_nl_name}')
        ret_msg = ''
        action_success=False

        if len(self.env.last_event.metadata['inventoryObjects']) > 0:
            holding_obj_id = self.env.last_event.metadata['inventoryObjects'][0]['objectId']
            if holding_obj_id is not None:
                log.warning(f'You can not open {target_nl_name} while holding an object. Please put it down first.')
                ret_msg = f'You can not open {target_nl_name} while holding an object. Please put it down first.'
                return ret_msg, action_success
        
        visible_objects = self.get_visible_obj_nl_names_from_last_event()
        if target_nl_name not in visible_objects:
            ret_msg = f'{target_nl_name} is not infront of you. '
            return ret_msg, action_success

        if target_nl_name not in self.high_name_to_obj_id_dict.keys():
            ret_msg = f'Cannot find {target_nl_name} to open. '
            return ret_msg, action_success
        
        obj_id = self.high_name_to_obj_id_dict[target_nl_name]

        # check is openable
        objects = self.env.last_event.metadata['objects']
        for o in objects:
            if o['objectId'] == obj_id:
                if not o['openable']:
                    ret_msg = f'{target_nl_name} can not be opened. '
                    return ret_msg, action_success
                if o['isOpen']:
                    ret_msg = f'{target_nl_name} is already opened. '
                    return ret_msg, action_success
                
        self.env.step(action="UnpausePhysicsAutoSim")
        self.env.step(
            action="OpenObject",
            objectId=obj_id,
            openness=1.0
        )
        self.env.step(action="PausePhysicsAutoSim")
        # if open action fails, back and try again
        max_move_step = 4

        if not self.env.last_event.metadata['lastActionSuccess']:
            log.warning(
                f"OpenObject action failed: {self.env.last_event.metadata['errorMessage']}, moving backward and trying again...")
            
            ret_msg = f"Open action failed. " 

            for move_step in range(max_move_step):
                # looks silly but it works
                log.warning(
                f"Try to move backward and try again... {move_step+1}/{max_move_step}")

                for _ in range (move_step):
                    self.env.step(action="MoveBack")

                self.env.step(
                    action="OpenObject",
                    objectId=obj_id,
                    openness=1.0
                )

                for _ in range (move_step):
                    self.env.step(action="MoveAhead")
                
                if self.env.last_event.metadata['lastActionSuccess']:
                    log.warning(f"OpenObject action success")
                    ret_msg = ''
                    action_success=True
                    break

            log.warning(f"OpenObject action failed: {self.env.last_event.metadata['errorMessage']}")
            return ret_msg, action_success
        else:
            action_success = True
            ret_msg = f'The {target_nl_name} is opened.'
        
        self.update_camera()
        return ret_msg, action_success
