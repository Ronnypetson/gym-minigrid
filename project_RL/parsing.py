import numpy as np


def linear_parse_observation_to_state(observation):
    ''' Parse to concatenation of one-hot encodings of observation.
        Each cell has the (OBJECT_IDX, COLOR_IDX, STATE) format.
        OBJECT_IDX: 11 possible values
        COLOR_IDX: 6 possible values
        STATE: 3 possible values
        direction: 4 possible values
    '''
    image = observation["image"]
    rows, cols, _ = image.shape
    feat_possible_values = {0: 11, 1: 6, 2: 3}
    num_directions = 4
    binary_features = []
    for row in range(rows):
        for column in range(cols):
            for feat, possible_vals in feat_possible_values.items():
                one_hot = [0 for _ in range(possible_vals)]
                one_hot[image[row][column][feat]] = 1
                binary_features += one_hot
    direction = [0 for _ in range(num_directions)]
    direction[observation["direction"]] = 1
    binary_features += direction
    binary_features = np.array(binary_features)
    return binary_features


def image_parse_observation_to_state(observation):
    """ Parse encoded observation to image format,
        oriented by the agent's direction.
    """
    img = observation['agent_fov_img']
    direction = observation['direction']
    img = np.rot90(img, k=direction, axes=(1, 0))
    return img


def parse_observation_to_state(observation):
    return tuple([tuple(observation["image"].flatten()),
                  observation["direction"]])
