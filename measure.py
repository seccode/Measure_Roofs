from matplotlib.widgets import Button
import argparse
from mapper import *

class Measurer:
    def __init__(self, image_folder):
        assert image_folder, "Image folder not specified"

        self.images = sorted(glob.glob(image_folder+"/*.png"), key=os.path.getctime)[:2]

        self.points = self.init_points(self.images)
    
    def init_points(self, images):
        return dict((image,{
                "Start": None,
                "End": None,
                "P_s": None,
                "P_e": None
            }) for image in images)
    
    def show_images(self):

        def init_plot():
            ax[0].imshow(plt.imread(self.images[0]),zorder=1)
            ax[1].imshow(plt.imread(self.images[1]),zorder=1)
        
        fig, ax = plt.subplots(2, 1, figsize=(15,10))
        init_plot()

        def add_point(event):
            table = {ax[0]: list(self.points.keys())[0],
                    ax[1]: list(self.points.keys())[1]}

            if event.inaxes in table:
                plot_position = table[event.inaxes]

                if not self.points[plot_position]["Start"]:
                    x, y = event.xdata, event.ydata
                    self.points[plot_position]["Start"] = (x, y)
                    event.inaxes.scatter(x, y, c='r', s=50, zorder=2)

                elif not self.points[plot_position]["End"]:
                    x, y = event.xdata, event.ydata
                    self.points[plot_position]["End"] = (x, y)
                    event.inaxes.scatter(x, y, c='b', s=50, zorder=2)
                    
                    event.inaxes.plot([self.points[plot_position]["Start"][0], x],
                        [self.points[plot_position]["Start"][1], y], c='g', zorder=2)

                else:
                    print("Points have already been selected")
            
            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', add_point)

        def print_measurement(event):
            print("Measuring")
            points = [
                list(self.points.values())[0]["Start"],
                list(self.points.values())[0]["End"],
                list(self.points.values())[1]["Start"],
                list(self.points.values())[1]["End"]
            ]
            assert not None in points, "Points have not been selected"

            mapper = Mapper()
            mapper.load_views(self.images)

            point_positions = []
            for i, view in enumerate(mapper.views):
                for j in range(i*2,i*2+2):
                    point_position = mapper.get_pointed_camera_position(
                        view["position"], points[j], view["image"].shape)
                    
                    point_positions.append(point_position)
            
            plt.close()
            mapper.plot_pointed_positions(point_positions)
            
            roof_points = []
            for i in range(2):
                roof_point = mapper.get_closest_position_between_views(
                    point_positions[i], point_positions[i+2])

                roof_points.append(roof_point)

            res = mapper.get_distance_between_positions(*roof_points)
            dist = mapper.meters_to_feet(res)

            print("Distance: {} ft.".format(dist))
        
        def clear_points(event):
            ax[0].clear()
            ax[1].clear()
            init_plot()
            self.points = self.init_points(self.images)

        axcut1 = plt.axes([0.88, 0.01, 0.1, 0.075])
        bcut1 = Button(axcut1, 'Get\nMeasurement', color='red', hovercolor='green')
        bcut1.on_clicked(print_measurement)

        axcut2 = plt.axes([0.88, 0.12, 0.1, 0.075])
        bcut2 = Button(axcut2, 'Clear\nPoints', color='red', hovercolor='green')
        bcut2.on_clicked(clear_points)

        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", "-images_folder", dest="images_folder")
    args = parser.parse_args()

    m = Measurer(args.images_folder)
    m.show_images()