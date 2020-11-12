import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import urllib.request
import json
from haversine import haversine
from mpl_toolkits.mplot3d import Axes3D
from sympy import Plane, Point3D, Line3D


EARTH_RADIUS = 6371000 # meters
MINLAT, MAXLAT = -90, 90
MINLON, MAXLON = -180, 180

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.degrees(np.arctan2(y, x))
    phi = np.degrees(np.arctan2(np.sqrt(x**2 + y**2), z))

    return (r, theta, phi)

def spherical_to_cartesian(r, theta, phi):
    _theta, _phi = np.radians(theta), np.radians(phi)

    x = r * np.sin(_theta) * np.cos(_phi)
    y = r * np.sin(_theta) * np.sin(_phi)
    z = r * np.cos(_theta)

    return (x, y, z)

def get_x_y_between_coords(lat, lng, ref):
    x = haversine((lat, lng), (ref["lat"], lng), unit='m')
    y = haversine((lat, lng), (lat, ref["lng"]), unit='m')

    direction_map = {True: 1, False: -1}

    return (x * direction_map[lat > ref["lat"]],
            y * direction_map[lng < ref["lng"]])

class Mapper:

    @staticmethod
    def get_ground_elevation_at_lat_lng(lat, lng):
        query_url = 'https://maps.googleapis.com/maps/api/elevation/json?locations=' + str(lat) + ',' + str(lng) + '&key=AIzaSyD0qmjMtTkXGx8Mv6m86t7CxiY6jibaWqE'
        r = urllib.request.urlopen(query_url)
        data = json.loads(r.read().decode(r.info().get_param('charset') or 'utf-8'))

        return data["results"][0]["elevation"]

    def get_camera_position_lat_lng(self, _lat, _lng, _dist, pitch, head):
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

    def get_camera_position(self, params, primary_view=None):
        _params = params.lstrip("images/").rstrip(".png").split("_")

        _lat, _lng, _dist = float(_params[0]), float(_params[1]), float(_params[3][:-1])

        alt, pitch, _head = float(_params[-1]), float(_params[6][:-1]), float(_params[5][:-1])
        head = -_head
        if _head < 0:
            _head += 360

        fov = float(_params[4][:-1]) * 2

        lat, lng = self.get_camera_position_lat_lng(_lat, _lng, _dist, pitch, _head)

        alt -= Mapper.get_ground_elevation_at_lat_lng(lat, lng)

        position = {
                    "lat": lat,
                    "lng": lng,
                    "x": 0,
                    "y": 0,
                    "z": alt,
                    "heading": head,
                    "pitch": 180 - pitch,
                    "fov": fov
                    }

        if primary_view:
            x, y = get_x_y_between_coords(lat, lng, primary_view["position"])
            position["x"] = x
            position["y"] = y

        return position
    
    def label_roof_pixels(self):

        def onclick(points, event):
            print((int(event.xdata), int(event.ydata)))
            points.append([int(event.xdata), int(event.ydata)])
        
        for i, view in enumerate(self.views):
            points = []
            print("\nView {}".format(i))
            fig = plt.figure(figsize=(12,10))
            _ = fig.canvas.mpl_connect("button_press_event", lambda e: onclick(points, e))
            plt.imshow(view["image"])
            plt.show()
            np.save("points/" + str(i), points)
            print("\n")

    def load_polygons(self, image_file):
        polygon_file = image_file.rstrip(".png") + ".json"
        if os.path.isfile(polygon_file):
            data = json.load(open(polygon_file))
            return data["shapes"]
        
        return None

    def load_views(self, image_list=None):
        self.views = []

        if image_list:
            images = image_list
        else:
            images = sorted(glob.glob("images/*.png"), key=os.path.getctime)

        for i, image in enumerate(images):
            self.views.append({
                "image": plt.imread(image),
                "polygons": self.load_polygons(image)
            })
            if i == 0:
                self.views[-1]["position"] = self.get_camera_position(image)
            else:
                self.views[-1]["position"] = self.get_camera_position(image, self.views[0])

        return
    
    def plot_3D_roof(self, faces):
        # Plot the 3D positions of the roof
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i, face in enumerate(faces):
            for pos in face:
                ax.scatter3D(pos[0],pos[1],pos[2],c='b')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
    
    def get_relative_frame_shape(self, camera_position, frame_shape):
        # Get the shape of frame formed by camera FOV, assuming center of
        # the frame a distance of 1 away from the camera position
        FOV = np.radians(camera_position["fov"])
        aspect_ratio = frame_shape[1] / frame_shape[0]

        relative_frame_shape = (
            2 * np.tan(FOV / 2),
            2 * np.tan((1 / aspect_ratio) * FOV / 2)
        )

        return relative_frame_shape

    def get_frame_relative_pixel_position(self, camera_position, pixel, frame_shape):
        # Get the position of the pixel relative to the center of the frame
        relative_frame_shape = self.get_relative_frame_shape(camera_position, frame_shape)

        pixel_w_pos = (pixel[0] / frame_shape[1]) * relative_frame_shape[0] - relative_frame_shape[0] / 2
        pixel_h_pos = ((frame_shape[0] - pixel[1]) / frame_shape[0]) * relative_frame_shape[1] - relative_frame_shape[1] / 2

        _pitch = np.radians(camera_position["pitch"] - 90)

        frame_relative_pixel_position = (
            pixel_h_pos * np.sin(_pitch),
            pixel_w_pos,
            pixel_h_pos * np.cos(_pitch)
        )

        return frame_relative_pixel_position

    def get_camera_relative_pixel_position(self, camera_position, pixel, frame_shape):
        # Get cartesian position of pixel assuming frame is orthogonal
        # to the camera position, with the center of the frame a distance
        # of 1 away from the camera position. Assumption is also made that
        # heading towards center of frame is 0, as this simplifies calculations

        frame_relative_pixel_position = self.get_frame_relative_pixel_position(
            camera_position, pixel, frame_shape)
        
        _pitch = np.radians(camera_position["pitch"] - 90)
        
        X = (np.cos(_pitch) + frame_relative_pixel_position[0])
        Y = -frame_relative_pixel_position[1]
        Z = -np.sin(_pitch) + frame_relative_pixel_position[2]

        return (X, Y, Z)

    def get_pointed_camera_position(self, camera_position, pixel, frame_shape):
        # Get the pitch and heading that point towards the pixel

        x, y, z = self.get_camera_relative_pixel_position(camera_position, pixel, frame_shape)

        _, theta, phi = cartesian_to_spherical(x, y, z)

        pointed_camera_position = camera_position.copy()
        pointed_camera_position["heading"] += theta
        pointed_camera_position["pitch"] = phi

        return pointed_camera_position

    def get_camera_position_at_z(self, camera_position, z):
        if z == camera_position['z']:
            return (camera_position['x'], camera_position['y'], z)

        dz = camera_position['z'] - z

        _pitch = np.radians(camera_position["pitch"] - 90)
        _heading = np.radians(camera_position["heading"])

        B = dz / np.tan(_pitch)
        X = B * np.cos(_heading)
        Y = B * np.sin(_heading)

        return [camera_position['x'] + X, camera_position['y'] + Y, z]

    def get_distance_between_positions(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + \
                        (p1[1] - p2[1])**2 + \
                        (p1[2] - p2[2])**2)
    
    def get_midpoint_between_positions(self, p1, p2):
        return ((p1[0] + p2[0]) / 2,
                (p1[1] + p2[1]) / 2,
                (p1[2] + p2[2]) / 2)
    
    def meters_to_feet(self, d):
        return d * 3.281

    def get_closest_position_between_view_lines(self, view_lines):
        min_dist = float("inf")
        closest_position = None

        for p1 in view_lines[0]:
            for p2 in view_lines[1]:
                dist = self.get_distance_between_positions(p1, p2)
                if dist < min_dist:
                    min_dist = dist
                    closest_position = self.get_midpoint_between_positions(p1, p2)

        return closest_position

    def plot_view_lines(self, view_lines):
        # Plot 3D camera view lines
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for view_line in view_lines:
            ax.plot(
                [view_line[0][0], view_line[-1][0]],
                [view_line[0][1], view_line[-1][1]],
                [view_line[0][2], view_line[-1][2]]
            )

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
            
    def get_closest_position_between_views(self, camera_position_1, camera_position_2):
        # Get the point that lies closest to both views
        view_lines = []
        for c in (camera_position_1, camera_position_2):
            view_points = np.array([self.get_camera_position_at_z(c,z) for z in np.linspace(c['z'],0,200)])

            view_lines.append(view_points)
        
        # self.plot_view_lines(view_lines)

        return self.get_closest_position_between_view_lines(view_lines)

    def plot_pointed_positions(self, pointed_positions):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i, position in enumerate(pointed_positions):
            end = self.get_camera_position_at_z(position, 0)

            ax.plot(
                [position['x'], end[0]],
                [position['y'], end[1]],
                [position['z'], end[2]],
                c='b'
            )

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
        return

    def get_3D_roof_positions(self):
        # Get camera positions that point to roof points
        for i, view in enumerate(self.views):
            face_views = []
            for polygon in view["polygons"]:
                face = []
                for point in polygon["points"]:
                    point_position = self.get_pointed_camera_position(
                            view["position"], point, view["image"].shape)

                    face.append(point_position)

                face_views.append(face)

            self.views[i]["face_views"] = face_views
            # self.plot_pointed_positions(pointed_camera_positions)

        # Get roof positions from pointed positions
        faces_positions = []
        for face1, face2 in zip(self.views[0]["face_views"], self.views[1]["face_views"]):
            face_positions = []
            for p1, p2 in zip(face1, face2):

                roof_position = self.get_closest_position_between_views(p1, p2)
                face_positions.append(roof_position)
            
            faces_positions.append(face_positions)

        return faces_positions

if __name__ == "__main__":
    mapper = Mapper()
    mapper.load_views()

    faces_positions = mapper.get_3D_roof_positions()
    mapper.plot_3D_roof(faces_positions)
