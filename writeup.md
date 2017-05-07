
## Advanced Lane Finding Project

**Creig Cavanaugh - March 2017**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/calibration_original.jpg "Original"
[image1]: ./output_images/calibration_calibrated.jpg "Undistorted"
[image2]: ./output_images/undistorted_image.jpg "Road Transformed"
[image3]: ./output_images/pipeline_result.png "Binary Example"
[image4a]: ./output_images/transform_source_points.jpg "Transform Source Points"
[image4b]: ./output_images/transform_destination_points.jpg "Transform Destination Points"
[image5]: ./output_images/curveplot.png "Fit Visual"
[image6]: ./output_images/video_snapshot.png "Output"
[video1]: ./video_output.mp4 "Video"
[code1]: ./adv_lane_det.py "Code"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 435 through 483 of the file called ![adv_lane_det.py][code1].  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

###Original Image
![alt text][image0]

###Undistorted Image
![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 99 through 146 in `adv_lane_det.py`).  Here's an example of my output for this step. In this example, the blue represents the thresholding derived from the saturation and lightness color space and the red represents the combined sobel gradient thresholds.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform appears in lines 185 through 219 in the file `adv_lane_det.py`. The code utilizes the `getPerspectiveTransform` and `warpPerspective` OpenCV functions, and takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
#Get Image Size
img_size = (img.shape[1], img.shape[0])

#Define Trapezoid Source Points for perspective transformation
top_v = 0.073
hor_v = 0.622
bot_v = 0.433

src = np.float32([[
    (int((img_size[0]/2)-(bot_v*img_size[0])),img_size[1]),
    (int((img_size[0]/2)-(top_v*img_size[0])),int(img_size[1]*hor_v)), 
    (int((img_size[0]/2)+(top_v*img_size[0])),int(img_size[1]*hor_v)), 
    (int((img_size[0]/2)+(bot_v*img_size[0])),img_size[1])
    ]])

dst = np.float32([
    [0, img_size[1]], 
    [0, 0],
    [img_size[0], 0],
    [img_size[0], img_size[1]]
    ])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 85, 720      | 0, 720        | 
| 546, 447      | 0, 0      |
| 733, 447     | 1280, 0      |
| 1194, 720      | 1280, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4a]
![alt text][image4b]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I implemented this step in lines 221 through 320 in my code in `adv_lane_det.py`.  I utilized the sliding window approach as described in the 'Finding the Lanes' section of the Advanced Lane Finding lesson, utilizing the numpy `polyfit` function to perform a least squares 2nd order polynomial fit.

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 308 through 363 in my code in `adv_lane_det.py`.  It is primarily based on the code example provided in the 'Measuring Curvature' section of the Advanced Lane Finding lesson, and I tuned the calculation by applying the following pixel to meter correction factors.
```
ym_per_pix = 38/720 # meters per pixel in y dimension
xm_per_pix = 3.65/1047 # meters per pixel in x dimension
```


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 379 through 424 in my code in `adv_lane_det.py` in the function `pipeline()`.  To gain further understanding of how the result was derived, I have superimposed the birds-eye view and lane detection outputs onto the image. Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

One of the early problems I had was the pipeline was picking up shadows from the left hand road barrier, which tricked the algorithm into thinking it was the left lane line in some instances, and resulted in a good amount of lane shifting. I experimented with using different color thresholding, and in the end primarily used the saturation and lightness channels from the HLS converted output to detect the lane lines, and tune out shadows and high contrast areas I was picking up from the left road barrier.

I noticed that just using the saturation and lightness filter worked well, but didn't always pick up farther distance dashed lines, so in the threshold pipeline I added a combination of directional and magnitude sobel thresholding.  By combining HL thresholding and sobel thresholding, I was able to capture both close and far lines, with a good amount of noise and shadow filtering.

I had created a version of code that implemented skipping the sliding window search after establishing where the lane lines were, and although it worked very well at the beginning of the video, I found it would eventually get out of sync and cause erratic results, so the final code does not implement it.  I think a future upgrade to this code would be to enable the functionality, but have a means to re-sync if needed.

I also experimented with holding past values and averaging the polynomials in order to clean up the outputs.  I found averaging worked, but noticed there was a bit more lag than I liked, so I did not include it in the final code.  I think in a future version, some of these techniques could generate backup or referee values, and can be used for short timeframes when the primary values fall out of a threshold.


