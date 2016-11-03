# SDC_LaneDetection
Assignment P1 for SDC Nanodegree


## Summary:

This implementation tries to locate lane markings in input images and overlays the information on the input image



## Implementation details:
Algorithm can be split into following sub-blocks:

1. Extract Region of interest to save computations. 
  - Trapezoid with base as image width and height nearly spanning till the middle of image
  - Base width and top width were arrived at empirically
2. Convert to grayscale to reduce further computations
3. Create a mask for yellow and white areas in the image as lanes are usually demarked by white or yellow lines
4. Extract yellow and white areas
5. Extract edges from these areas using canny algorithm
6. Remove noise effect by blurring out canny edge image with a kernel of size 3
7. Find lines in this filtered image
8. Split the lines into left and right lanes using the slope of these lines
9. Fit a lane marking line passing through left and right set of lines we obtained from Hough transform
	- I have tried linear regression to fit the line passing through all points in a particular lane
    - Other methods could include passing a line through top and bottom points in a particular lane
10. Overlay the lane markings on input image
  
### Note: The algorithm informs with a print spew if it failed to detect one/both lane markings but continues to find lane markings in subsequent images




## Tested with  
- test_images
- solidWhiteRight.mp4
- solidYellowLeft.mp4
- challenge.mp4 
