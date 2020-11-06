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

def get_image_position(f, primary_view=None):
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
                "pitch": pitch - 90,
                "fov": fov
                }

    if primary_view:
        x, y = get_x_y(lat, lng, primary_view["position"])
        position["x"] = x
        position["y"] = y

    return position

def get_pitch_and_heading_between_positions(pos1, pos2):
    dx, dy, dz = pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]
    heading = np.arctan2(dz, dx)
    pitch = np.arctan2(np.sqrt(dz**2 + dx**2), dy) + np.pi
    return (pitch, heading)

def get_distance_between_positions(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0])**2 + \
                   (pos1[1] - pos2[1])**2 + \
                   (pos1[2] - pos2[2])**2)

def get_midpoint_between_positions(pos1, pos2):
    return [(p1 + p2) / 2 for p1, p2 in zip(pos1, pos2)]

def get_position_at_z(camera_position, z=0):
    # print("x: {}\ny: {}\nz: {}\npitch: {}\nheading: {}\n".format(
        # camera_position['x'], camera_position['y'], camera_position['z'], camera_position['pitch'], camera_position['head']))

    if z == camera_position['z']:
        return (camera_position['x'], camera_position['y'], camera_position['z'])

    Y = np.cos(np.radians(camera_position["head"])) * (camera_position['z'] - z) / \
        np.tan(np.radians(-camera_position["pitch"]))
    
    X = Y * np.tan(np.radians(camera_position["head"]))

    return (camera_position['x'] + X, camera_position['y'] + Y, z)

def get_closest_position_between_ranges(range1,range2):
    position = None
    min_dist = float("inf")
    for p1 in range1:
        for p2 in range2:
            dist = get_distance_between_positions(p1,p2)
            if dist < min_dist:
                min_dist = dist
                position = get_midpoint_between_positions(p1,p2)

    return position

def plot_3D(range1, range2, midpoint):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for p1, p2 in zip(range1,range2):
            ax.scatter3D(p1[0],p1[1],p1[2],c='b')
            ax.scatter3D(p2[0],p2[1],p2[2],c='r')
        
        ax.scatter3D(midpoint[0],midpoint[1],midpoint[2],c='g')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

def get_pixel_world_position(camera_pos1, camera_pos2):
    c1 = np.array([camera_pos1['x'], camera_pos1['y'], camera_pos1['z']])
    c2 = np.array([camera_pos2['x'], camera_pos2['y'], camera_pos2['z']])

    range1 = list(map(lambda z: get_position_at_z(camera_pos1,z),np.linspace(c1[-1],0)))
    range2 = list(map(lambda z: get_position_at_z(camera_pos2,z),np.linspace(c2[-1],0)))

    print("\tC1\n-----------------\nx: {}\ny: {}\nz: {}\n".format(*c1))
    print("\tC2\n-----------------\nx: {}\ny: {}\nz: {}\n".format(*c2))

    world_position = get_closest_position_between_ranges(range1,range2)

    plot_3D(range1, range2, world_position)

    return world_position
    

def get_pitch_and_heading(camera_position, pixel_pos, frame_shape):
    rel_frame_width = 2 * np.tan(np.radians(camera_position["fov"] / 2))
    rel_frame_height = 2 * np.tan(np.radians((
        frame_shape[0] / frame_shape[1]) * camera_position["fov"] / 2))

    rel_pixel_x_pos = (((frame_shape[1] - pixel_pos[0]) / frame_shape[1]) - 0.5) * rel_frame_width
    rel_pixel_r_pos = (((frame_shape[0] - pixel_pos[1]) / frame_shape[0]) - 0.5) * rel_frame_height

    _pitch = np.radians(camera_position["pitch"])

    pixel_pos = [
        rel_pixel_x_pos,
        np.cos(_pitch) + np.sin(_pitch) * rel_pixel_r_pos,
        np.sin(_pitch) - np.cos(_pitch) * rel_pixel_r_pos
    ]

    theta = np.degrees(np.arctan2(pixel_pos[1], pixel_pos[0]))
    phi = np.degrees(np.arctan2(np.sqrt(pixel_pos[0]**2 + pixel_pos[1]**2), -pixel_pos[2]))

    new_heading = camera_position["head"] + (theta - 90)
    new_pitch = phi - 90

    print(camera_position["pitch"])
    print(new_pitch)

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
        (2236, 981),
        (2026, 984),
        (1740, 1314),
        (990, 658),
        (1229, 357),
        (1483, 435)
    ]

    img2_pts = [
        (1580, 1458),
        (1285, 1103),
        (1044, 1107),
        (1787, 479),
        (2069, 435),
        (2292, 739)
    ]

    def onclick(event):
        print((int(event.xdata), int(event.ydata)))

    # fig = plt.figure()
    # cid = fig.canvas.mpl_connect("button_press_event", onclick)
    # plt.imshow(image_map["view_1"]["img"])
    # plt.show()
    # a

    frame_shape = image_map["view_0"]["img"].shape
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
    
    def show_roof():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for pos in roof_positions:
            ax.scatter3D(pos[0],pos[1],pos[2],c='b')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
    
    print(roof_positions)
    show_roof()
