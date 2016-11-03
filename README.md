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

## Running the code
To run the code, open P1.ipynb using jupyter notebook and execute cells one after the other or all in one go. 

After the execution is complete, output for test images would be created in the same folder as "test_images/" but with file names prepended with "lanes_". Please delete all files with name starting with "lanes_" before running the algorithm again

The code could also be run using python. In this case open P1.py instead of P1.ipynb and execute using 
$ python P1.py

The python code was used for development and P1.ipynb would be considered for submission. It could be enhanced further with arguments to decide whether to operate in video mode or image mode and also clean up output files before subsequent execution. This task will be taken up if time permits at a later stage

## Tested with  
- test_images
- solidWhiteRight.mp4
- solidYellowLeft.mp4
- challenge.mp4 
