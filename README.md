# Measure Roofs
This is a tool that can be used to measure roof dimensions or any 3D line distance from Google Earth images

# How to use
Add the [Google Earth Photo Taker](https://chrome.google.com/webstore/detail/google-earth-photo-taker/nigkfjnfbnlnkjkgoknmdkobpngidilo/related) chrome extension to your browser.

Using the chrome extension in Google Earth Web, take 2 photos of a location from different angles (do not rename the downloaded image files):
Angle 1             |  Angle 2
:-------------------------:|:-------------------------:
![](https://i.ibb.co/jrZVW95/42-94082118-76-43288172-271-45986219a-166-55002227d-35y-89-64533778h-65-52781164t-0r-340.png)  |  ![](https://i.ibb.co/Gt4Vwhy/42-94072053-76-43280389-276-63764533a-156-06850922d-35y-163-53830087h-67-52204743t-0r-336.png)

Place the images in the following folder structure:
```
.
|--mapper.py
|--measure.py
|--images/
   |--<image-1>
   |--<image-2>
```
Run the following command from the root of the repository:
```
python measure.py --f images/
```

The following matplotlib figure will open:             |  Click on the images to choose start and end points for the line you want to measure:
:-------------------------:|:-------------------------:
![](https://i.ibb.co/q9rZhmV/preselect.png) | ![](https://i.ibb.co/SttbXJX/postselect.png)

Make sure that the red and blue points for each image correspond to each other. Then click the "Get Measurement" button; the distance of the line will output to the terminal. Click the "Clear Points" button to remove the selected points from the images.

The angles from the Google Earth camera position to the selected pixels will be generated, then matched with the corresponding angle from the other image to find the closest point between the angles. This locates both points in 3D space and allows for a simple distance calculation to find the measurement between the points
![](https://i.ibb.co/Jzd7WcX/Screen-Shot-2020-11-13-at-4-28-14-PM.png)
