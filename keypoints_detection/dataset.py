# import tensorflow as tf

def get_keypoints(kp_file_path):
    with open(kp_file_path) as f:
        contents = f.read()

    contents = contents.split('{')[1]
    contents = contents.split('}')[0]

    lines = contents.strip().split('\n')
    points = [
        tuple(map(float, kp.split(' '))) for kp in lines]

    return points
