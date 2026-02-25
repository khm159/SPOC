import os, json, re
import string
import subprocess
from collections import defaultdict
import copy

ALFRED_OBJS = {'Cart', 'Potato', 'Faucet', 'Ottoman', 'CoffeeMachine', 'Candle', 'CD', 'Pan', 'Watch',
            'HandTowel', 'SprayBottle', 'BaseballBat', 'CellPhone', 'Kettle', 'Mug', 'StoveBurner', 'Bowl',
            'Toilet', 'DiningTable', 'Spoon', 'TissueBox', 'Shelf', 'Apple', 'TennisRacket', 'SoapBar',
            'Cloth', 'Plunger', 'FloorLamp', 'ToiletPaperHanger', 'CoffeeTable', 'Spatula', 'Plate', 'Bed',
            'Glassbottle', 'Knife', 'Tomato', 'ButterKnife', 'Dresser', 'Microwave', 'CounterTop',
            'GarbageCan', 'WateringCan', 'Vase', 'ArmChair', 'Safe', 'KeyChain', 'Pot', 'Pen', 'Cabinet',
            'Desk', 'Newspaper', 'Drawer', 'Sofa', 'Bread', 'Book', 'Lettuce', 'CreditCard', 'AlarmClock',
            'ToiletPaper', 'SideTable', 'Fork', 'Box', 'Egg', 'DeskLamp', 'Ladle', 'WineBottle', 'Pencil',
            'Laptop', 'RemoteControl', 'BasketBall', 'DishSponge', 'Cup', 'SaltShaker', 'PepperShaker',
            'Pillow', 'Bathtub', 'SoapBottle', 'Statue', 'Fridge', 'Sink'}
ALFRED_PICK_OBJ = {'KeyChain', 'Potato', 'Pot', 'Pen', 'Candle', 'CD', 'Pan', 'Watch', 'Newspaper', 'HandTowel',
                'SprayBottle', 'BaseballBat', 'Bread', 'CellPhone', 'Book', 'Lettuce', 'CreditCard', 'Mug',
                'AlarmClock', 'Kettle', 'ToiletPaper', 'Bowl', 'Fork', 'Box', 'Egg', 'Spoon', 'TissueBox',
                'Apple', 'TennisRacket', 'Ladle', 'WineBottle', 'Cloth', 'Plunger', 'SoapBar', 'Pencil',
                'Laptop', 'RemoteControl', 'BasketBall', 'DishSponge', 'Cup', 'Spatula', 'SaltShaker',
                'Plate', 'PepperShaker', 'Pillow', 'Glassbottle', 'SoapBottle', 'Knife', 'Statue', 'Tomato',
                'ButterKnife', 'WateringCan', 'Vase'}
AFLRED_OPEN_OBJ = {'Safe', 'Laptop', 'Fridge', 'Box', 'Microwave', 'Cabinet', 'Drawer'}
ALFRED_SLICE_OBJ = {'Potato', 'Lettuce', 'Tomato', 'Apple', 'Bread'}
ALFRED_TOGGLE_OBJ = {'Microwave', 'DeskLamp', 'FloorLamp', 'Faucet'}
ALFRED_RECEP = {'ArmChair', 'Safe', 'Cart', 'Ottoman', 'Pot', 'CoffeeMachine', 'Desk', 'Cabinet', 'Pan',
                'Drawer', 'Sofa', 'Mug', 'StoveBurner', 'SideTable', 'Toilet', 'Bowl', 'Box', 'DiningTable',
                'Shelf', 'ToiletPaperHanger', 'CoffeeTable', 'Cup', 'Plate', 'Bathtub', 'Bed', 'Dresser',
                'Fridge', 'Microwave', 'CounterTop', 'Sink', 'GarbageCan', 'BathtubBasin', 'SinkBasin',
                'HandTowelHolder', 'PaintingHanger', 'Pan', 'Pot', 'TowelHolder', 'Safe', 'LaundryHamper', 
                'TVStand','Toaster'}
AI2THOR_CUTTING_TOOL = {'Knife','ButterKnife'}
AI2THOR_HEAT_SOURCE = {'StoveKnob','Microwave'}
AI2THOR_HEAT_COKER = {'Pan','Pot'}
AI2THOR_BREAKING_TOOL = {'BaseballBat','TennisRacket','Knife','Statue'}

AFLRED_RECEP_MOVABLE = {'Bowl','Box','Cup','Mug','Plate','Pan','Pot'}

STATIC_RECEPTACLES = set(ALFRED_RECEP) - set(AFLRED_RECEP_MOVABLE)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def ithor_name_to_natural_word(w):
    # e.g., RemoteController -> remote controller
    if w == 'CD':
        return w
    else:
        return re.sub(r"(\w)([A-Z])", r"\1 \2", w).lower()

def natural_word_to_ithor_name(w):
    # e.g., floor lamp -> FloorLamp
    if w == 'CD':
        return w
    else:
        return ''.join([string.capwords(x) for x in w.split()])

# make a dictionary {id: name}
def make_name_id_dict(object_ids):
    """
    objectIds : ['window|7|0', ...]
    name : [Window (number), ...]
    """
    object_ids.sort(reverse=True)
    
    name_id_dict = {}
    counter_dict = {}

    for obj_id in object_ids:
        base_name = obj_id.split('|')[0]
        if base_name not in counter_dict:
            counter_dict[base_name] = 1

        name_id_dict[obj_id] = f"{base_name} ({counter_dict[base_name]})"
        counter_dict[base_name] += 1

    return name_id_dict

# 'window|7|0' --> window (1) 
def convert_id_to_nlname(object_id, name_dict):
    return name_dict[object_id]

# window (1) --> 'window|7|0'
def convert_nlname_to_id(object_name, name_dict):
    for k, v in name_dict.items():
        if v == object_name:
            return k

# chair_14j5913 -> chair 1
def name_id_dict_sim2nl(object_list, dict_id):
    init_obs_sim2nl = []
    action_obj = []
    for k, v in dict_id.items():
        print(k,v)
        init_obs_sim2nl.append(v)
        for obj in object_list:
            if k == obj:
                action_obj.append(v)
    action_obj_seperated = copy.deepcopy(action_obj)
    action_obj = group_objects_by_name(action_obj)

    return action_obj_seperated, action_obj

# chair 1 -> chair_14j5913 
def name_id_dict_nl2sim(object_str, dict_id):
    init_obs_nl2sim = []
    action_obj = []
    object_list = ungroup_objects(object_str)

    for k, v in dict_id.items():
        init_obs_nl2sim.append(k)
        for obj in object_list:
            if v == obj:
                action_obj.append(k)

    return init_obs_nl2sim, action_obj

# chair (1), chair (2) -> chair (1, 2)
def group_objects_by_name(object_ids):
    grouped_objects = defaultdict(list)
    object_list = []

    for obj_id in object_ids:
        match = re.match(r"([a-zA-Z]+)\s*\((\d+)\)", obj_id)
        if match:
            obj_name = match.group(1).strip()
            obj_number = match.group(2).strip()
            grouped_objects[obj_name].append(obj_number)

    for obj_name, obj_numbers in grouped_objects.items():
        obj = f"{obj_name} ({', '.join(obj_numbers)})"
        object_list.append(obj)
    return object_list

# chair (1, 2) -> chair (1), chair (2)
def ungroup_objects(grouped_objects):
    object_ids = []
    match = re.match(r"(\w+)\s*\(([\d,\s]+)\)", grouped_objects)

    if match:
            obj_name = match.group(1)
            obj_numbers = match.group(2).replace(" ", "").split(",")
            
            for number in obj_numbers:
                object_ids.append(f"{obj_name} ({number})")
    else:
        object_ids.append(grouped_objects)

    return object_ids

# Output objects except for those in closed receptacles
def obs_partial_objs(objs):
    check_recep = []
    init_recep = []
    init_obs = []

    for obj in objs:
        if obj['receptacle']:
            recep_info = {
                'recep_name': obj['name'],
                'openable': obj['openable'],
                'isOpen': obj['isOpen'],
                'receptacleobjectIds': obj['receptacleObjectIds']
            }
            check_recep.append(recep_info)

    for recep in check_recep:
        if recep['openable'] and not recep['isOpen'] and recep['receptacleobjectIds']:
            init_recep.extend(recep['receptacleobjectIds'])

    # objects
    filtered_objects = [obj['name'] for obj in objs if obj['objectId'] not in init_recep if obj['pickupable'] is True]

    # receptacles
    recep = [obj['name'] for obj in objs if obj['receptacle']]

    init_obs = filtered_objects + recep

    return init_obs

def obs_partial_recep(objs):
    recep_info = set()
    init_obs = []

    for obj in objs:
        if obj['receptacle']:
            recep_info.add(obj['name'])
    
    for o in recep_info:
        recep = o.split('_')[0]
        if recep in STATIC_RECEPTACLES:
            init_obs.append(o)

    return init_obs

def recall_working_memory(working_memory, target_obj):
    # 빈공간 제거, Capitalize 하여 성공률 높임 
    target_obj_class = target_obj.split("(")[0]
    target_obj_class = natural_word_to_ithor_name(target_obj_class)
    # try:
    #     target_obj_class_backup = copy.deepcopy(target_obj_class)
    #     target_obj_class = target_obj_class.capitalize()
    # except:
    #     target_obj_class = target_obj_class_backup
    #     pass
    if target_obj_class in working_memory:
        target_obj_loc_list = working_memory[target_obj_class]
        messages = []
        for location_info in target_obj_loc_list:
            target_obj_id = location_info['id'] # 실제 opbject 이름 ex) Apple (1)
            nl_location_obj = location_info['location_obj'] # 실제 location 이름 ex) DiningTable (1)
            if 'agent' in nl_location_obj: 
                message = f'You are holding {target_obj} {target_obj_id}.'
            elif 'drop' in nl_location_obj:
                message = f'You dropped {target_obj} due to fail to put it on the receptacle. You need to find it again.'
            else:
                if nl_location_obj:
                    message = f'You saw {target_obj_id} near {nl_location_obj}.'
                else:
                    message = f'You saw {target_obj_id}.'
            messages.append(message)
        obs_text = ' '.join(messages)
    else:
        obs_text = f'You have not seen {target_obj} before.'
    return obs_text

def update_working_memory(working_memory, obj_name, location):
    """
    Alfred is a single room 
    - working_memory : dict, working memory dictionaray 
    - obj_name : str, object name ex) 'Potato (1)'
    - location : str, location name ex) 'DiningTable (1)'
    """
    ### working memory: {
    # 'class_name': 
    #       [
    #           {'obj_name': , 'location':}
    #       ]
    # }
    # print("update working memory")
    # print("[input] obj_name : ", obj_name)
    # print("[input] location : ", location)
    obj_class = obj_name.split("(")[0]
    obj_class = natural_word_to_ithor_name(obj_class)
    # print("[input] obj_class (natural_word_to_ithor_name): ", obj_class)
    if obj_class in ALFRED_RECEP:
        if not obj_class in AFLRED_RECEP_MOVABLE:
            return working_memory
    if obj_class not in working_memory:
        # print("obj_class not in working_memory, create new list")
        working_memory[obj_class] = []
    updated = False
    for i, entry in enumerate(working_memory[obj_class]):
        # print("Update exist object location!")
        if entry['id'] == obj_name:
            # print("i-th object [{}] location update [{}] to [{}] ".format(
            #     obj_name,
            #     working_memory[obj_class][i]['location_obj'],
            #     location
            # ))
            working_memory[obj_class][i]['location_obj'] = location # update
            updated = True

    if not updated:
        if not obj_class in list(working_memory.keys()):
            working_memory[obj_class] = []
        # print("Add new object location!", obj_name, location)
        working_memory[obj_class].append(
            {'id': obj_name, 'location_obj': location}
        )
    return working_memory

def update_working_memory_open(working_memory, obj_name, location):
    """
    Alfred is a single room 
    - working_memory : dict, working memory dictionaray 
    - obj_name : str, object name ex) 'Potato (1)'
    - location : str, location name ex) 'DiningTable (1)'
    """
    obj_class = obj_name.split("(")[0]
    # obj_class = obj_class.replace(" ", "")
    obj_class = natural_word_to_ithor_name(obj_class)
    if obj_class in ALFRED_RECEP:
        if not obj_class in AFLRED_RECEP_MOVABLE:
            return working_memory
    if obj_class not in working_memory:
        working_memory[obj_class] = []
    is_prv_seen = False

    for entry in working_memory[obj_class]:
        if entry['id'] == obj_name:
            is_prv_seen = True

    if not is_prv_seen:
        working_memory[obj_class].append(
            {'id': obj_name, 'location_obj': location}
        )
    return working_memory


def delete_obj_from_working_memory(working_memory, obj_name, location):
    obj_class = obj_name.split("(")[0]
    # obj_class = obj_class.replace(" ", "")
    obj_class = natural_word_to_ithor_name(obj_class)

    if obj_class in ALFRED_RECEP:
        if not obj_class in AFLRED_RECEP_MOVABLE:
            return working_memory
    
    if obj_class not in working_memory:
        return working_memory # error! not in working memory
    
    is_prv_seen = False

    for entry in working_memory[obj_class]:
        if entry['id'] == obj_name:
            is_prv_seen = True

    if not is_prv_seen:
        working_memory[obj_class].append(
            {'id': obj_name, 'location_obj': location}
        )

    # delete target object from working memory
    # sliced object must be deleted 
    delete_index= None
    for i, entry in enumerate(working_memory[obj_class]):
        if entry['id'] == obj_name:
            is_prv_seen = True
            delete_index = i
    # delete list element 
    if delete_index is not None:
        del working_memory[obj_class][delete_index]
    return working_memory
