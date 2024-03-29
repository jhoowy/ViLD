import os
import os.path as osp
import numpy as np
import pickle5 as pickle
import json
from tqdm import tqdm

from lvis import LVIS

import torch
import clip
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse

BASE_CLASSES = (
    'aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol', 
    'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna', 
    'apple', 'apron', 'aquarium', 'armband', 'armchair', 'artichoke', 
    'trash_can', 'ashtray', 'asparagus', 'atomizer', 'avocado', 'award', 
    'awning', 'baby_buggy', 'basketball_backboard', 'backpack', 'handbag', 
    'suitcase', 'bagel', 'ball', 'balloon', 'bamboo', 'banana', 'Band_Aid', 
    'bandage', 'bandanna', 'banner', 'barrel', 'barrette', 'barrow', 'baseball_base', 
    'baseball', 'baseball_bat', 'baseball_cap', 'baseball_glove', 'basket', 
    'basketball', 'bat_(animal)', 'bath_mat', 'bath_towel', 'bathrobe', 
    'bathtub', 'battery', 'bead', 'bean_curd', 'beanbag', 'beanie', 'bear', 
    'bed', 'bedspread', 'cow', 'beef_(food)', 'beer_bottle', 'beer_can', 
    'bell', 'bell_pepper', 'belt', 'belt_buckle', 'bench', 'beret', 'bib', 
    'bicycle', 'visor', 'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 
    'birdbath', 'birdcage', 'birdhouse', 'birthday_cake', 'black_sheep', 
    'blackberry', 'blackboard', 'blanket', 'blazer', 'blender', 'blinker', 
    'blouse', 'blueberry', 'boat', 'bobbin', 'bobby_pin', 'boiled_egg', 
    'deadbolt', 'bolt', 'book', 'bookcase', 'booklet', 'boot', 'bottle', 
    'bottle_opener', 'bouquet', 'bow_(decorative_ribbons)', 'bow-tie', 
    'bowl', 'bowler_hat', 'box', 'suspenders', 'bracelet', 'brassiere', 
    'bread-bin', 'bread', 'bridal_gown', 'briefcase', 'broccoli', 'broom', 
    'brownie', 'brussels_sprouts', 'bucket', 'bull', 'bulldog', 'bullet_train', 
    'bulletin_board', 'bullhorn', 'bun', 'bunk_bed', 'buoy', 'bus_(vehicle)', 
    'business_card', 'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabin_car', 
    'cabinet', 'cake', 'calculator', 'calendar', 'calf', 'camcorder', 
    'camel', 'camera', 'camera_lens', 'camper_(vehicle)', 'can', 'can_opener', 
    'candle', 'candle_holder', 'candy_cane', 'walking_cane', 'canister', 
    'canoe', 'cantaloup', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino', 
    'car_(automobile)', 'railcar_(part_of_a_train)', 'identity_card', 
    'card', 'cardigan', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 
    'carton', 'cash_register', 'cast', 'cat', 'cauliflower', 'cayenne_(spice)', 
    'CD_player', 'celery', 'cellular_telephone', 'chair', 'chandelier', 
    'cherry', 'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'crisp_(potato_chip)', 
    'chocolate_bar', 'chocolate_cake', 'choker', 'chopping_board', 'chopstick', 
    'Christmas_tree', 'slide', 'cigarette', 'cigarette_case', 'cistern', 
    'clasp', 'cleansing_agent', 'clip', 'clipboard', 'clock', 'clock_tower', 
    'clothes_hamper', 'clothespin', 'coaster', 'coat', 'coat_hanger', 
    'coatrack', 'cock', 'coconut', 'coffee_maker', 'coffee_table', 'coffeepot', 
    'coin', 'colander', 'coleslaw', 'pacifier', 'computer_keyboard', 'condiment', 
    'cone', 'control', 'cookie', 'cooler_(for_food)', 'cork_(bottle_plug)', 
    'corkscrew', 'edible_corn', 'cornet', 'cornice', 'corset', 'costume', 
    'cowbell', 'cowboy_hat', 'crab_(animal)', 'cracker', 'crate', 'crayon', 
    'crescent_roll', 'crib', 'crock_pot', 'crossbar', 'crow', 'crown', 
    'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch', 'cub_(animal)', 
    'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup', 'cupboard', 'cupcake', 
    'curtain', 'cushion', 'dartboard', 'deck_chair', 'deer', 'dental_floss', 
    'desk', 'diaper', 'dining_table', 'dish', 'dish_antenna', 'dishrag', 
    'dishtowel', 'dishwasher', 'dispenser', 'Dixie_cup', 'dog', 'dog_collar', 
    'doll', 'dolphin', 'domestic_ass', 'doorknob', 'doormat', 'doughnut', 
    'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit', 'dresser', 
    'drill', 'drum_(musical_instrument)', 'duck', 'duckling', 'duct_tape', 
    'duffel_bag', 'dumpster', 'eagle', 'earphone', 'earring', 'easel', 
    'egg', 'egg_yolk', 'eggbeater', 'eggplant', 'refrigerator', 'elephant', 
    'elk', 'envelope', 'eraser', 'fan', 'faucet', 'Ferris_wheel', 'ferry', 
    'fighter_jet', 'figurine', 'file_cabinet', 'fire_alarm', 'fire_engine', 
    'fire_extinguisher', 'fire_hose', 'fireplace', 'fireplug', 'fish', 
    'fish_(food)', 'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 
    'flap', 'flashlight', 'flip-flop_(sandal)', 'flipper_(footwear)', 
    'flower_arrangement', 'flute_glass', 'foal', 'folding_chair', 'food_processor', 
    'football_(American)', 'footstool', 'fork', 'forklift', 'freight_car', 
    'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice', 'frying_pan', 
    'garbage_truck', 'garden_hose', 'gargle', 'garlic', 'gazelle', 'gelatin', 
    'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture', 'glass_(drink_container)', 
    'globe', 'glove', 'goat', 'goggles', 'golf_club', 'golfcart', 'goose', 
    'grape', 'grater', 'gravestone', 'green_bean', 'green_onion', 'grill', 
    'grizzly', 'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet', 
    'hairpin', 'ham', 'hamburger', 'hammer', 'hammock', 'hamster', 'hair_dryer', 
    'hand_towel', 'handcart', 'handkerchief', 'handle', 'hat', 'veil', 
    'headband', 'headboard', 'headlight', 'headscarf', 'headstall_(for_horses)', 
    'heart', 'heater', 'helicopter', 'helmet', 'highchair', 'hinge', 'hog', 
    'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'horse', 'hose', 
    'hot_sauce', 'hummingbird', 'polar_bear', 'icecream', 'ice_maker', 
    'igniter', 'iPod', 'iron_(for_clothing)', 'ironing_board', 'jacket', 
    'jam', 'jar', 'jean', 'jeep', 'jersey', 'jet_plane', 'jewelry', 'jumpsuit', 
    'kayak', 'kettle', 'key', 'kilt', 'kimono', 'kitchen_sink', 'kite', 
    'kitten', 'kiwi_fruit', 'knee_pad', 'knife', 'knob', 'ladder', 'ladle', 
    'ladybug', 'lamb_(animal)', 'lamp', 'lamppost', 'lampshade', 'lantern', 
    'lanyard', 'laptop_computer', 'latch', 'legging_(clothing)', 'Lego', 
    'lemon', 'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb', 
    'lime', 'lion', 'lip_balm', 'lizard', 'log', 'lollipop', 'speaker_(stero_equipment)', 
    'loveseat', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)', 
    'mandarin_orange', 'manger', 'manhole', 'map', 'marker', 'mashed_potato', 
    'mask', 'mast', 'mat_(gym_equipment)', 'mattress', 'measuring_cup', 
    'measuring_stick', 'meatball', 'medicine', 'melon', 'microphone', 
    'microwave_oven', 'milk', 'minivan', 'mirror', 'mitten', 'mixer_(kitchen_tool)', 
    'money', 'monitor_(computer_equipment) computer_monitor', 'monkey', 
    'motor', 'motor_scooter', 'motorcycle', 'mound_(baseball)', 'mouse_(computer_equipment)', 
    'mousepad', 'muffin', 'mug', 'mushroom', 'musical_instrument', 'napkin', 
    'necklace', 'necktie', 'needle', 'nest', 'newspaper', 'newsstand', 
    'nightshirt', 'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 
    'oar', 'oil_lamp', 'olive_oil', 'onion', 'orange_(fruit)', 'orange_juice', 
    'ostrich', 'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 
    'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas', 'palette', 
    'pan_(for_cooking)', 'pancake', 'paper_plate', 'paper_towel', 'parachute', 
    'parakeet', 'parasail_(sports)', 'parasol', 'parka', 'parking_meter', 
    'parrot', 'passenger_car_(part_of_a_train)', 'passport', 'pastry', 
    'pea_(food)', 'peach', 'peanut_butter', 'pear', 'peeler_(tool_for_fruit_and_vegetables)', 
    'pelican', 'pen', 'pencil', 'penguin', 'pepper', 'pepper_mill', 'perfume', 
    'person', 'pet', 'pew_(church_bench)', 'phonograph_record', 'piano', 
    'pickle', 'pickup_truck', 'pie', 'pigeon', 'pillow', 'pineapple', 
    'pinecone', 'pipe', 'pita_(bread)', 'pitcher_(vessel_for_liquid)', 
    'pizza', 'place_mat', 'plate', 'platter', 'pliers', 'pocketknife', 
    'poker_(fire_stirring_tool)', 'pole', 'polo_shirt', 'pony', 'pop_(soda)', 
    'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato', 
    'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel', 
    'printer', 'projectile_(weapon)', 'projector', 'propeller', 'pumpkin', 
    'puppy', 'quilt', 'rabbit', 'racket', 'radiator', 'radio_receiver', 
    'radish', 'raft', 'raincoat', 'ram_(animal)', 'raspberry', 'razorblade', 
    'reamer_(juicer)', 'rearview_mirror', 'receipt', 'recliner', 'record_player', 
    'reflector', 'remote_control', 'rhinoceros', 'rifle', 'ring', 'robe', 
    'rocking_chair', 'rolling_pin', 'router_(computer_equipment)', 'rubber_band', 
    'runner_(carpet)', 'plastic_bag', 'saddle_(on_an_animal)', 'saddle_blanket', 
    'saddlebag', 'sail', 'salad', 'salami', 'salmon_(fish)', 'salsa', 
    'saltshaker', 'sandal_(type_of_shoe)', 'sandwich', 'saucer', 'sausage', 
    'scale_(measuring_instrument)', 'scarf', 'school_bus', 'scissors', 
    'scoreboard', 'screwdriver', 'scrubbing_brush', 'sculpture', 'seabird', 
    'seahorse', 'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark', 
    'shaving_cream', 'sheep', 'shield', 'shirt', 'shoe', 'shopping_bag', 
    'shopping_cart', 'short_pants', 'shoulder_bag', 'shovel', 'shower_head', 
    'shower_curtain', 'signboard', 'silo', 'sink', 'skateboard', 'skewer', 
    'ski', 'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'sled', 'sleeping_bag', 
    'slipper_(footwear)', 'snowboard', 'snowman', 'snowmobile', 'soap', 
    'soccer_ball', 'sock', 'sofa', 'solar_array', 'soup', 'soupspoon', 
    'sour_cream', 'spatula', 'spectacles', 'spice_rack', 'spider', 'sponge', 
    'spoon', 'sportswear', 'spotlight', 'squirrel', 'stapler_(stapling_machine)', 
    'starfish', 'statue_(sculpture)', 'steak_(food)', 'steering_wheel', 
    'step_stool', 'stereo_(sound_system)', 'stirrup', 'stool', 'stop_sign', 
    'brake_light', 'stove', 'strainer', 'strap', 'straw_(for_drinking)', 
    'strawberry', 'street_sign', 'streetlight', 'suit_(clothing)', 'sunflower', 
    'sunglasses', 'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 
    'sweatband', 'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 
    'sword', 'table', 'table_lamp', 'tablecloth', 'tag', 'taillight', 
    'tank_(storage_vessel)', 'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 
    'tape_measure', 'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 
    'teacup', 'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth', 
    'telephone_pole', 'television_camera', 'television_set', 'tennis_ball', 
    'tennis_racket', 'thermometer', 'thermos_bottle', 'thermostat', 'thread', 
    'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil', 
    'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven', 
    'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush', 
    'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel', 
    'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light', 
    'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'tray', 
    'tricycle', 'tripod', 'trousers', 'truck', 'trunk', 'turban', 'turkey_(food)', 
    'turtle', 'turtleneck_(clothing)', 'typewriter', 'umbrella', 'underwear', 
    'urinal', 'urn', 'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 
    'vest', 'videotape', 'volleyball', 'waffle', 'wagon', 'wagon_wheel', 
    'walking_stick', 'wall_clock', 'wall_socket', 'wallet', 'automatic_washer', 
    'watch', 'water_bottle', 'water_cooler', 'water_faucet', 'water_jug', 
    'water_scooter', 'water_ski', 'water_tower', 'watering_can', 'watermelon', 
    'weathervane', 'webcam', 'wedding_cake', 'wedding_ring', 'wet_suit', 
    'wheel', 'wheelchair', 'whipped_cream', 'whistle', 'wig', 'wind_chime', 
    'windmill', 'window_box_(for_plants)', 'windshield_wiper', 'windsock', 
    'wine_bottle', 'wine_bucket', 'wineglass', 'blinder_(for_horses)', 
    'wok', 'wooden_spoon', 'wreath', 'wrench', 'wristband', 'wristlet', 
    'yacht', 'yogurt', 'yoke_(animal_equipment)', 'zebra', 'zucchini')

A = 4
C = 0.6
M = 0.05

parser = argparse.ArgumentParser(description='Calculate embedding weights for ViLD-SL')
parser.add_argument('--data_root', default='/data/project/rw/lvis_v1', type=str)
parser.add_argument('--save_dir', default='img_embeddings', type=str)
parser.add_argument('--ann_save_dir', default='annotations/lvis_v1_train_embed.json', type=str)
parser.add_argument('--text_embed_path', default='text_embeddings/lvis_cf.pickle', type=str)
args = parser.parse_args()

data_root = args.data_root
ann_file = osp.join(data_root, 'annotations/lvis_v1_train.json')
save_dir = osp.join(data_root, args.save_dir)
ann_save_dir = osp.join(data_root, args.ann_save_dir)
text_embed_path = osp.join(data_root, args.text_embed_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(text_embed_path ,'rb') as f:
    text_embed = pickle.load(f)
text_embed = torch.from_numpy(text_embed).to(device)

coco = LVIS(ann_file)

cats = coco.dataset["categories"]
cats = [cat for cat in cats if cat['name'] in BASE_CLASSES]
cat_ids = [cat['id'] for cat in cats]
cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}

img_ids = coco.get_img_ids()

correct_count = 0
incorrect_count = 0

json_result = dict()
chart = np.zeros(867)
for i in tqdm(img_ids):
    img_info = coco.load_imgs([i])[0]
    filename = img_info['coco_url'].replace('http://images.cocodataset.org/', '')

    img = Image.open(osp.join(data_root, filename))
    outname = '.'.join((filename.split('.')[0], 'pickle'))

    ann_ids = coco.get_ann_ids(img_ids=[i], cat_ids=cat_ids)
    ann_info = coco.load_anns(ann_ids)
    with open(osp.join(save_dir, outname), 'rb') as f:
        embeddings = pickle.load(f)
    for ann in ann_info:
        ann_id = ann['id']
        gt_label = cat2label[ann['category_id']]

        if ann.get('ignore', False):
            continue
            
        x1, y1, w, h = ann['bbox']
        inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
        inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
        if inter_w * inter_h == 0:
            continue
        if ann['area'] <= 0 or w < 1 or h < 1:
            continue

        im_embed = torch.from_numpy(embeddings[ann_id]).to(device)
        scores = (100.0 * im_embed @ text_embed.T).softmax(dim=-1)
        pred_label = int(torch.argmax(scores, dim=-1))
        pred_score = float(scores[0, gt_label].cpu())

        sorted_score = torch.sort(scores[0]).values
        rank = 866 - int((sorted_score == pred_score).nonzero()[0])
        chart[rank] += 1

        # Previous version
        # emb_weight = max(0.1, 1 / rank)

        # y = A / (1 + B*C^-x) + M
        B = (A - M) / M
        emb_weight = A / (1 + B*(C**(-rank))) + M

        if pred_label == gt_label:
            correct_count += 1
            emb_weight = 1
        else:
            incorrect_count += 1

        json_result[ann_id] = emb_weight

with open(ann_save_dir, 'w') as out:
    json.dump(json_result, out)

print(f"Corrects: {correct_count}, incorrects: {incorrect_count}")
plt.bar(np.arange(867), chart)
plt.tight_layout()
plt.savefig('emb_scores.png')
