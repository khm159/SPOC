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
ALFRED_PICK_OBJ = {'KeyChain', 'Potato', 'PotatoSliced', 'Pot', 'Pen', 'Candle', 'CD', 'Pan', 'Watch', 'Newspaper', 'HandTowel',
                'SprayBottle', 'BaseballBat', 'Bread','BreadSliced', 'CellPhone', 'Book', 'Lettuce', 'LettuceSliced', 'CreditCard', 
                'Mug', 'PaperTowelRoll','ScrubBrush', 'TeddyBear', 'Towel',
                'AlarmClock', 'Kettle', 'ToiletPaper', 'Bowl', 'Fork', 'Box', 'Egg', 'Spoon', 'TissueBox',
                'Apple', 'AppleSliced','TennisRacket', 'Ladle', 'WineBottle', 'Cloth', 'Plunger', 'SoapBar', 'Pencil',
                'Laptop', 'RemoteControl', 'BasketBall', 'DishSponge', 'Cup', 'Spatula', 'SaltShaker',
                'Plate', 'PepperShaker', 'Pillow', 'Glassbottle', 'SoapBottle', 'Knife', 'Statue', 'Tomato', 'TomatoSliced',
                'ButterKnife', 'WateringCan', 'Vase', 'Boots', 'Bottle', 'Dumbbell', 'HandTowel'}
AFLRED_OPEN_OBJ = {
    'Safe', 'Laptop', 'Fridge', 'Box', 'Microwave', 'Cabinet', 'Drawer', 'Blinds', 
    'Book', 'Kettle', 'ShowerCurtain', 'ShowerDoor','Toilet'}
ALFRED_SLICE_OBJ = {'Potato', 'Lettuce', 'Tomato', 'Apple', 'Bread'}
ALFRED_TOGGLE_OBJ = {
    'Candle', 'CellPhone', 'CoffeeMachine','DeskLamp', 'Faucet', 'FloorLamp', 
    'Laptop', 'LightSwitch', 'Microwave','ShowerHead', 'StoveKnob','StoveBurner', 
    'Television', 'Toaster'
    }

ALFRED_RECEP = {'ArmChair', 'Safe', 'Cart', 'Ottoman', 'Pot', 'CoffeeMachine', 'Desk', 'Cabinet', 'Pan',
                'Drawer', 'Sofa', 'Mug', 'StoveBurner', 'SideTable', 'Toilet', 'Bowl', 'Box', 'DiningTable',
                'Shelf', 'ToiletPaperHanger', 'CoffeeTable', 'Cup', 'Plate', 'Bathtub', 'Bed', 'Dresser',
                'Fridge', 'Microwave', 'CounterTop', 'Sink', 'GarbageCan', 'BathtubBasin', 'SinkBasin',
                'HandTowelHolder', 'PaintingHanger', 'Pan', 'Pot', 'TowelHolder', 'Safe', 'LaundryHamper', 
                'TVStand','Toaster'}
ALFRED_FILLABLE_OBJ = {
    'Bottle', 'Bowl', 'Cup', 'Mug', 'HousePlant', 'Kettle', 'Pot', 'WateringCan', 'WineBottle'
}

ALFRED_BREAKABLE_OBJ = {
    'Bottle', 'Bowl', 'CellPhone', 'Cup', 'Egg', 'Laptop', 'Mirror', 'Mug', 'Plate', 'ShowerDoor', 'Statue', 'Television', 'Vase', 'Window', 'WineBottle'
}

AI2THOR_NOT_PICKABLE_OBJ = {
    'ArmChair','Bathtub','BathtubBasin','Bed','Cabinet', 'CoffeeMachine', 'Chair', 'Blinds',
    'CoffeeTable', 'CounterTop', 'Curtains', 'Desk', 'DeskLamp', 'Desktop', 'DiningTable',
    'DogBed', 'Drawer', 'Dresser', 'Faucet', 'Floor', 'FloorLamp', 'Footstool', 'Fridge',
    'GarbageCan', 'GarbageBag', 'HandTowelHolder', 'HousePlant', 'LaundryHamper', 'LightSwitch',
    'Microwave', 'Mirror', 'Ottoman', 'Painting', 'Poster', 'RoomDecor', 'Safe', 'Shelf', 'ShelvingUnit',
    'ShowerCurtain', 'ShowerDoor', 'ShowerGlass', 'ShowerHead', 'SideTable', 'Sink', 'SinkBasin',
    'Sofa', 'Stool', 'StoveBurner', 'StoveKnob', 'TableTopDecor', 'TargetCircle', 'Television',
    'Toaster', 'Toilet', 'ToiletPaperHanger', 'TowelHolder', 'TVStand', 'VacuumCleaner',
    'Window'
}

AFLRED_RECEP_MOVABLE = {'Bowl','Box','Cup','Mug','Plate','Pan','Pot'}
ALFRED_RECEP_LIQUID_POUR = {'Bowl', 'Cup', 'Mug'} 
ALFRED_RECEP_LIQUID_FILL = {'Bowl', 'Cup', 'Mug', 'Pot','Kettle','WateringCan'} # container for liquid 
ALFRED_LIQUID_TYPE = {'Wine', 'Coffee', 'Water'} # three liquid type 

ALFRED_LIQUID_SOURCE = {
    'Wine':['WineBottle'], 
    'Water':['Bottle', 'Bowl', 'Kettle'],
    'Coffee':['Kettle']
} # liquid source to pour into container

STATIC_RECEPTACLES = set(ALFRED_RECEP) - set(AFLRED_RECEP_MOVABLE)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_task_json(task):
    '''
    load preprocessed json from disk
    '''
    json_path = os.path.join('alfred/data/json_2.1.0', task['task'], 'pp',
                             'ann_%d.json' % task['repeat_idx'])
    with open(json_path) as f:
        data = json.load(f)
    return data


def print_gpu_usage(msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """

    def query(field):
        return (subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
             '--format=csv,nounits,noheader'],
            encoding='utf-8'))

    def to_int(result):
        return int(result.strip().split('\n')[0])

    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used / total
    print('\n' + msg, f'{100 * pct:2.1f}% ({used} out of {total})')


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


def find_indefinite_article(w):
    # simple rule, not always correct
    w = w.lower()
    if w[0] in ['a', 'e', 'i', 'o', 'u']:
        return 'an'
    else:
        return 'a'

# make a dictionary {id: name}
def make_name_id_dict(object_ids):
    object_ids.sort(reverse=True)
    
    name_id_dict = {}
    counter_dict = {}

    for obj_id in object_ids:
        base_name = obj_id.split('(')[0].split('_')[0]

        if base_name not in counter_dict:
            counter_dict[base_name] = 1

        name_id_dict[obj_id] = f"{base_name} ({counter_dict[base_name]})"
        counter_dict[base_name] += 1

    return name_id_dict

# chair_14j5913 -> chair 1
def name_id_dict_sim2nl(object_list, dict_id):
    init_obs_sim2nl = []
    action_obj = []
    for k, v in dict_id.items():
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

def read_txt_file(file_path):
    with open(file_path) as file:
        txt_content = file.read()
    return txt_content