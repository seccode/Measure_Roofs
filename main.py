import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import urllib.request
import json
from haversine import haversine
from mpl_toolkits.mplot3d import Axes3D

EARTH_RADIUS = 6371000 # meters
MINLAT, MAXLAT = -90, 90
MINLON, MAXLON = -180, 180

def get_ground_elevation_at_lat_lng(lat, lng):
    query_url = 'https://maps.googleapis.com/maps/api/elevation/json?locations=' + str(lat) + ',' + str(lng) + '&key=AIzaSyD0qmjMtTkXGx8Mv6m86t7CxiY6jibaWqE'
    r = urllib.request.urlopen(query_url)
    data = json.loads(r.read().decode(r.info().get_param('charset') or 'utf-8'))

    return data["results"][0]["elevation"]

def get_x_y(lat, lng, ref):
    x = haversine((lat, lng), (lat, ref["lng"]), unit='m')
    y = haversine((lat, lng), (ref["lat"], lng), unit='m')
    direction_map = {True: 1, False: -1}
    return (x * direction_map[lng > ref["lng"]],
            y * direction_map[lat > ref["lat"]])

def get_camera_lat_lng(_lat, _lng, _dist, pitch, head):
    delta = np.sin(np.radians(pitch)) * _dist / EARTH_RADIUS
    theta = np.radians((head + 180) % 360)

    phi1, lambda1 = np.radians(_lat), np.radians(_lng)

    phi2 = np.arcsin(
        np.sin(phi1) * np.cos(delta) + \
        np.cos(phi1) * np.sin(delta) * np.cos(theta)
    )

    lambda2 = lambda1 + \
        np.arctan2(
            np.sin(theta) * np.sin(delta) * np.cos(phi1),
            np.cos(delta) - np.sin(phi1) * np.sin(phi2)
        )

    lng = np.degrees(lambda2)

    if lng < MINLON or lng > MAXLON:
        lambda2 = ((lambda2 + 3 * np.pi) % (2 * np.pi)) - np.pi
        lng = np.degrees(lambda2)

    return (np.degrees(phi2), lng)

def get_image_position(f, top_view=None):
    params = f.lstrip("imgs/").rstrip(".png").split("_")

    _lat, _lng, _dist = float(params[0]), float(params[1]), float(params[3][:-1])

    alt, pitch, head = float(params[-1]), float(params[6][:-1]), float(params[5][:-1])
    if head < 0:
        head += 360

    fov = float(params[4][:-1]) * 2

    lat, lng = get_camera_lat_lng(_lat, _lng, _dist, pitch, head)

    alt -= get_ground_elevation_at_lat_lng(lat, lng)

    position = {
                "lat": lat,
                "lng": lng,
                "x": 0,
                "y": 0,
                "z": alt,
                "head": head,
                "pitch": pitch,
                "fov": fov
                }
    if top_view:
        x, y = get_x_y(lat, lng, top_view["position"])
        position["x"] = x
        position["y"] = y

    return position

def get_pitch_and_heading_between_positions(pos1, pos2):
    dx, dy, dz = pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]
    heading = np.arctan2(dz, dx)
    pitch = np.arctan2(np.sqrt(dz**2 + dx**2), dy) + np.pi
    return (pitch, heading)

def get_ground_position(camera_pos):
    ground_dist = np.sin(np.radians(camera_pos["pitch"]))
    g_x = camera_pos['x'] + ground_dist * np.sin(np.radians(camera_pos["head"]))
    g_y = camera_pos['y'] + ground_dist * np.cos(np.radians(camera_pos["head"]))

    return (g_x, g_y, 0)

def get_position_ranges(camera_pos, ground_pos):
    return [np.linspace(start, end, 300)
        for start, end in zip((camera_pos['x'], camera_pos['y'], camera_pos['z']), ground_pos)]

def get_distance_between_positions(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0])**2 + \
                   (pos1[1] - pos2[1])**2 + \
                   (pos1[2] - pos2[2])**2)

def get_midpoint_between_positions(pos1, pos2):
    return [(p1 + p2) / 2 for p1, p2 in zip(pos1, pos2)]

def get_pixel_world_position(camera_pos1, camera_pos2):
    ground_pos1, ground_pos2 = \
        (get_ground_position(c) for c in (camera_pos1, camera_pos2))

    ranges1, ranges2 = (get_position_ranges(c, g)
        for c, g in zip([camera_pos1, camera_pos2], [ground_pos1, ground_pos2]))

    print(ranges1)
    min_val = float("inf")
    world_position = None

    for i in range(300):
        pos1 = (ranges1[0][i], ranges1[1][i], ranges1[2][i])
        pos2 = (ranges2[0][i], ranges2[1][i], ranges2[2][i])
        dist = get_distance_between_positions(pos1, pos2)
        if dist < min_val:
            min_val = dist
            world_position = get_midpoint_between_positions(pos1, pos2)

    return world_position

def get_pitch_and_heading(camera_position, pixel_pos, frame_shape):
    rel_frame_width = 2 * np.tan(np.radians(camera_position["fov"]))
    rel_frame_height = 2 * np.tan(np.radians((
        frame_shape[0] / frame_shape[1]) * camera_position["fov"]))

    """
    Camera Position = (0, 0, 0)
    Frame Center Position = (0, cos(pitch), sin(pitch))
    Frame Pixel Position (FPP) = (rel_pixel_x_pos,
                            cos(pitch) + sin(pitch) * rel_pixel_z_pos,
                            sin(pitch) + cos(pitch) * rel_pixel_z_pos)
    
    Camera to Pixel Pitch = arctan2(sqrt(FPP.x**2 + FPP.z**2), FPP.y) + pi
    Camera to Pixel Heading = arctan2(FPP.z, FPP.x)
    """

    rel_pixel_x_pos = ((pixel_pos[0] - frame_shape[0]) / frame_shape[0]) * rel_frame_width
    rel_pixel_z_pos = ((pixel_pos[1] - frame_shape[1]) / frame_shape[1]) * rel_frame_height

    _pitch = np.radians(camera_position["pitch"])

    pixel_pos = (
        rel_pixel_x_pos,
        np.cos(_pitch) + np.sin(_pitch) * rel_pixel_z_pos,
        np.sin(_pitch) + np.cos(_pitch) * rel_pixel_z_pos
    )

    new_pitch = np.arctan2(np.sqrt(pixel_pos[0]**2 + pixel_pos[2]**2), pixel_pos[1]) * np.pi
    new_heading = camera_position["head"] + np.arctan2(pixel_pos[2], pixel_pos[0])

    return (new_pitch, new_heading)

def get_position_toward_pixel(image_info, pixel_pos):
    frame_shape = image_info["img"].shape
    assert pixel_pos["x"] in range(frame_shape[1]), "Pixel x value out of range"
    assert pixel_pos["y"] in range(frame_shape[0]), "Pixel y value out of range"

    camera_position = image_info["position"]

    adj_camera_position = camera_position.copy()

    new_pitch, new_heading = get_pitch_and_heading(
        camera_position, pixel_pos, frame_shape)

    adj_camera_position["pitch"] = new_pitch
    adj_camera_position["head"] = new_heading

    return adj_camera_position

def get_images():
    ret = {}
    images = sorted(glob.glob("imgs/*"), key=os.path.getctime)

    for i, image in enumerate(images):
        view_id = "view_" + str(i)
        ret[view_id] = {
            "img": cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        }
        if i == 0:
            ret[view_id]["position"] = get_image_position(image)
        else:
            ret[view_id]["position"] = get_image_position(image, ret["view_0"])

    return ret

if __name__ == "__main__":
    image_map = get_images()

    img1_pts = [
        (427, 399),
        (577, 245),
        (741, 330),
        (1027, 635),
        (836, 584),
        (635, 780)
    ]

    img2_pts = [
        (938, 300),
        (1096, 224),
        (1232, 369),
        (1042, 752),
        (841, 573),
        (660, 635)
    ]

    frame_shape = image_map["view_0"]["img"].shape

    plt.imshow(image_map["view_0"]["img"])

    for p in img1_pts:
        plt.scatter(p[0], p[1], c='r')
    plt.show()

    roof_positions = []
    
    for p1, p2 in zip(img1_pts, img2_pts):
        c1 = image_map["view_0"]["position"].copy()
        new_pitch, new_heading = get_pitch_and_heading(c1, p1, frame_shape)
        c1["pitch"], c1["head"] = new_pitch, new_heading

        c2 = image_map["view_1"]["position"].copy()
        new_pitch, new_heading = get_pitch_and_heading(c2, p2, frame_shape)
        c2["pitch"], c2["head"] = new_pitch, new_heading

        pos = get_pixel_world_position(c1, c2)
        roof_positions.append(pos)

    print(roof_positions)
