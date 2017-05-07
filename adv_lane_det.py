#Advanced Lane Detection
#Creig Cavanaugh - March 2017
#

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
	# Apply the following steps to img
	# 1) Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# 2) Take the derivative in x or y given orient = 'x' or 'y'
	if (orient == 'x'):
	    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, None, sobel_kernel)
	else:
	    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, None, sobel_kernel)

	# 3) Take the absolute value of the derivative or gradient
	abs_sobel = np.absolute(sobel)

	# 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

	# 5) Create a mask of 1's where the scaled gradient magnitude 
	        # is > thresh_min and < thresh_max
	binary_output = np.zeros_like(scaled_sobel)
	binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

	# 6) Return this mask as your binary_output image
	return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
	# Apply the following steps to img
	# 1) Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# 2) Take the gradient in x and y separately
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, None, sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, None, sobel_kernel)
	# 3) Calculate the magnitude 
	abs_sobelxy= np.sqrt(np.add(np.square(sobelx), np.square(sobely)))
	# 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
	scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))

	# 5) Create a binary mask where mag thresholds are met
	binary_output = np.zeros_like(scaled_sobel)
	binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

	# 6) Return this mask as your binary_output image
	return binary_output
    

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	# Apply the following steps to img
	# 1) Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# 2) Take the gradient in x and y separately
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, None, sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, None, sobel_kernel)
	# 3) Take the absolute value of the x and y gradients
	abs_sobelx = np.absolute(sobelx)
	abs_sobely = np.absolute(sobely)
	# 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
	scaled_sobel = np.arctan2(abs_sobely, abs_sobelx)
	# 5) Create a binary mask where direction thresholds are met
	binary_output = np.zeros_like(scaled_sobel)
	binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
	# 6) Return this mask as your binary_output image
	return binary_output


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
	# 1) Convert to HLS color space
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	H = hls[:,:,0]
	L = hls[:,:,1]
	S = hls[:,:,2]
	# 2) Apply a threshold to the S channel
	binary_output = np.zeros_like(S)
	binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
	# 3) Return a binary image of threshold result
	return binary_output

#### Define the Threshold Pipeline ####
def threshold_pipeline(img, s_thresh=(95, 255), l_thresh=(40, 255), mag_thres=(40,85), dir_thres=(0.8, 1.2), sx_thresh=(30, 100)):
	img = np.copy(img)
	# Convert to HSV color space and separate the V channel
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
	l_channel = hsv[:,:,1]
	s_channel = hsv[:,:,2]

	# Threshold color channel (Saturation)
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

	# Threshold color channel (Lighness)
	l_binary = np.zeros_like(l_channel)
	l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

	combined_color = np.zeros_like(s_binary)
	combined_color[((s_binary  == 1) & (l_binary == 1))] = 1

	#Directional threshold
	dir_binary = dir_threshold(img, sobel_kernel=5, thresh=dir_thres)
	mag_binary = mag_thresh(img, sobel_kernel=5, mag_thresh=mag_thres)

	combined_mag_dir = np.zeros_like(dir_binary)
	combined_mag_dir[((mag_binary == 1) & (dir_binary == 1))] = 1

	# Note: Enable the following for write-up images, otherwise disable
	# # Stack each channel
	# # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
	# # be beneficial to replace this channel with something else.
	# color_binary = np.dstack((combined_mag_dir, np.zeros_like(dir_binary), combined_color))

	# # Plot the result
	# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	# f.tight_layout()

	# ax1.imshow(img)
	# ax1.set_title('Original Image', fontsize=40)

	# ax2.imshow(color_binary)
	# ax2.set_title('Pipeline Result', fontsize=40)
	# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	# plt.savefig('output_images/pipeline_result.png')


	combined_binary = np.zeros_like(dir_binary)
	combined_binary[((combined_mag_dir == 1) | (combined_color == 1))] = 1

	return combined_binary


##### Define Main Pipeline ######
def pipeline(img, mtx, dist, s_thresh=(170, 255), sx_thresh=(20, 100)):
	img = np.copy(img)

	img_size = (img.shape[1], img.shape[0])

	#Correct for distortion
	img_corrected = cv2.undistort(img, mtx, dist, None, mtx)

	##Sample images for writeup
	#cv2.imwrite('output_images/original_image.jpg',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
	#cv2.imwrite('output_images/undistorted_image.jpg',cv2.cvtColor(img_corrected, cv2.COLOR_RGB2BGR))


	#Define Trapezoid Source Points for perspective transformation
	top_v = 0.073
	hor_v = 0.622
	bot_v = 0.433

	vertices = np.array([[
		(int((img_size[0]/2)-(bot_v*img_size[0])),img_size[1]),
		(int((img_size[0]/2)-(top_v*img_size[0])),int(img_size[1]*hor_v)), 
		(int((img_size[0]/2)+(top_v*img_size[0])),int(img_size[1]*hor_v)), 
		(int((img_size[0]/2)+(bot_v*img_size[0])),img_size[1])
		]], dtype=np.int32)

	##Make a copy of the corrected image and superimpose the transformation boundary
	##Note: Only enable / needed for project write-up, otherwise disable
	#image_boundary = np.copy(cv2.cvtColor(img_corrected, cv2.COLOR_RGB2BGR))
	#cv2.polylines(image_boundary, vertices, True, [0,0,255],4)
	#cv2.imwrite('output_images/P05_transform_source_points.jpg',image_boundary)

	#Create a thresholded binary image
	threshold_binary = threshold_pipeline(img_corrected);


	###Perform a perspective transform
	# For source points, using points as defined via top_v, hor_v, and bot_v parameters
	src = np.float32([[
		(int((img_size[0]/2)-(bot_v*img_size[0])),img_size[1]),
		(int((img_size[0]/2)-(top_v*img_size[0])),int(img_size[1]*hor_v)), 
		(int((img_size[0]/2)+(top_v*img_size[0])),int(img_size[1]*hor_v)), 
		(int((img_size[0]/2)+(bot_v*img_size[0])),img_size[1])
		]])

	#print(src)

	# For destination points, I'm using the full image size
	dst = np.float32([
		[0, img_size[1]], 
		[0, 0],
		[img_size[0], 0],
		[img_size[0], img_size[1]]
		])

	#print(dst)

	# Given src and dst points, calculate the perspective transform matrix
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)

	# Warp the image using OpenCV warpPerspective()
	# Note - This is primarily only used to provide visual on the video - the binary_warped is used by the rest of the pipeline
	warped = cv2.warpPerspective(img_corrected, M, img_size)

	##Enable to write warped image to disk
	#warped_boundary = cv2.warpPerspective(image_boundary, M, img_size)
	#cv2.imwrite('output_images/P05_transform_destination_points.jpg', warped_boundary)
	
	# Warp binary image
	binary_warped = cv2.warpPerspective(threshold_binary, M, img_size)

	##Find the Lines
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# Set the width of the windows +/- margin
	margin = 105  #was 100

	#Identify lane-line pixels and fit their positions with a polynomial
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	window_img = np.zeros_like(out_img)


	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)


	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base


	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
	    win_y_low = binary_warped.shape[0] - (window+1)*window_height
	    win_y_high = binary_warped.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
	    # Draw the windows on the visualization image
	    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
	    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
	    # Identify the nonzero pixels in x and y within the window
	    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
	    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
	    # Append these indices to the lists
	    left_lane_inds.append(good_left_inds)
	    right_lane_inds.append(good_right_inds)
	    # If you found > minpix pixels, recenter next window on their mean position
	    if len(good_left_inds) > minpix:
	        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
	    if len(good_right_inds) > minpix:        
	        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	#Calculate the radius of curvature of the lane and the position of the vehicle with respect to center
	# Define y-value where we want radius of curvature
	y_eval = np.max(ploty)
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 38/720 # meters per pixel in y dimension
	xm_per_pix = 3.65/1047 # meters per pixel in x dimension

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

	mid_pixel = (binary_warped.shape[1]/2)
	mid_lane = (((right_fitx[719] - left_fitx[719])/2)+ left_fitx[719])
	print("mid lane:", mid_lane)
	delta_center = (mid_pixel - mid_lane)
	print("delta center:", delta_center)
	delta_center_m = (xm_per_pix*delta_center)


	#Create moving average for distance from center
	delta_center_ma.append(delta_center_m)

	if len(delta_center_ma) > 40:
		delta_center_ma.pop(0)

	delta_center_mean = sum(delta_center_ma) / float(len(delta_center_ma))


	print('delta from center: ', delta_center_mean)

	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

	#Create moving average for left and right curvature
	left_curverad_ma.append(left_curverad)

	if len(left_curverad_ma) > curvature_ma:
		left_curverad_ma.pop(0)

	left_curverad_mean = sum(left_curverad_ma) / float(len(left_curverad_ma))

	right_curverad_ma.append(right_curverad)

	if len(right_curverad_ma) > curvature_ma:
		right_curverad_ma.pop(0)

	right_curverad_mean = sum(right_curverad_ma) / float(len(right_curverad_ma))

	avg_curverad = int((left_curverad_mean + right_curverad_mean) /2)

	# Now our radius of curvature is in meters
	print(left_curverad, 'm', right_curverad, 'm', avg_curverad, 'm')


	#Generate plot with left and right curve lines
	#Note - This is used as a visualization for the video
	cvt_out = cv2.cvtColor(out_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
	plt.imshow(cvt_out)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	#plt.show()
	plt.savefig('output_images/curveplot.png')
	plt.close()


	#Create image that has been warped back onto the original image and plotted to identify the lane boundaries

	# Create an image to draw the lines on
	warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))


	color_warp[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	color_warp[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (img_corrected.shape[1], img_corrected.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(img_corrected, 1, newwarp, 0.3, 0)
	#plt.imshow(result)

	curvetext = ("Radius of Curvature = " +  str(avg_curverad) + "(m)")
	cv2.putText(result, curvetext, (50, 50 ), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))

	if (delta_center_mean > 0):
		centertext = ("Vehicle is {:2.2f}".format(abs(delta_center_mean)) + "(m) right of center")
	else:
		centertext = ("Vehicle is {:2.2f}".format(abs(delta_center_mean)) + "(m) left of center")
	cv2.putText(result, centertext, (50, 100 ), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))

	#Add in top-down transformation
	scaled_warped = cv2.resize(warped,None,fx=.2, fy=.2, interpolation = cv2.INTER_CUBIC)

	x_offset=result.shape[1]-(scaled_warped.shape[1]+20)
	y_offset=20
	result[y_offset:y_offset+scaled_warped.shape[0], x_offset:x_offset+scaled_warped.shape[1]] = scaled_warped

	img_poly = cv2.imread('output_images/curveplot.png')
	scaled_poly = cv2.resize(img_poly ,None,fx=.5, fy=.5, interpolation = cv2.INTER_CUBIC)
	
	x_offset=result.shape[1]-(scaled_poly.shape[1]+20)
	y_offset=180
	result[y_offset:y_offset+scaled_poly.shape[0], x_offset:x_offset+scaled_poly.shape[1]] = scaled_poly

	return result


def process_image(image):
	returned_img = pipeline(image, mtx, dist)
	return returned_img

################################

#### Calibrate Camera ####
nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        cv2.imshow('img', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()


# Test undistortion on an image
img = cv2.imread('camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

dst = cv2.undistort(img, mtx, dist, None, mtx)

cv2.imwrite('output_images/calibration_original.jpg',img)
cv2.imwrite('output_images/calibration_calibrated.jpg',dst)


####
# Setup moving averages
left_curverad_ma = []
right_curverad_ma = []
delta_center_ma = []

curvature_ma = 50

####################
#Test sample images on pipeline
# images = glob.glob('test_images/test*.jpg')

#Step through the list and search for chessboard corners
#for idx, fname in enumerate(images):
#     img = mpimg.imread(fname)
#     returned_img = pipeline(img, mtx, dist)
#     plt.imshow(returned_img)
#     plt.show()



#clip1 = VideoFileClip("cut3.mp4")
#output = 'output_images/video3_c.mp4'

clip1 = VideoFileClip("project_video.mp4")
output = 'output_images/video_output.mp4'

clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
clip.write_videofile(output, audio=False)
