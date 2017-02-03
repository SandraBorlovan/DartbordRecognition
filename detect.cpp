#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <math.h>
#include <typeinfo>
#include <stdlib.h>
#include <fstream>
#include <string>

#define RADIUS_LENGTH  50
#define RADIUS_LOW     20
#define RADIUS_HIGH    85
#define CENTER_THR     235
#define RADIUS         63
#define MAGN_THR       62
#define HOUGH_THR      240
#define SQUARES_THR    50
#define RECT_THR	   30
#define BLUR_THR       0

using namespace std;
using namespace cv;

struct square {
  double x;
  double y;
  double width;
  double height;
} ;

struct Circle {
	Point center;
	double radius;
	bool merged;
};

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;


/**
  * Displaying data images
  */
void displayImage(Mat image);

/**
  * Displaying original images
  */
void displayOriginalImage(Mat image);

/**
  * Draw rectangles on a given image
  *
  * @squares vector of rectangles
  * @image   image to draw the rectangles on
  */
void drawSquaresFromArray(square squares[], Mat image, int length);

void drawSquaresFromVector(vector<Rect> squares, Mat image);

/**
  * Draw squares from cicles center points
  *
  * @image         image to draw on
  * @hough_space   hough space for circles
  * @thr           threshold for selecting a center
  * @radius        for determining the size of the squares
  */
void drawSquaresFromCenter(Mat image, Mat hough_space, double thr, double radius);

/**
  * Draw circles from cicles center points
  *
  * @image         image to draw on
  * @hough_space   hough space for circles
  * @thr           threshold for selecting a center
  * @radius        draw circles of size of this radius
  */
void displayCircles(Circle circles[], int length, Mat image);

/**
  * Remove the squares that have an average value
  * in the hough space less then a threshold
  *
  * @hough_   space hough space for circles
  * @detected vector of detected squares
  * @thr      threshold for the average value
  */
void thrSquares(Mat hough_space, vector<Rect>& detected, double thr);

int thrRect(Mat hough_space, square squares[], double thr, int length);

/**
  * Do intersection between two rectangles
  *
  * @return true or false
  */
bool doRectIntersect(Rect square_1, Rect square_2);

/**
  * Do intersection between two squares
  *
  * @return true or false
  */
bool doSquareIntersect(square square_1, Rect square_2);

/**
  * Calculate circle intersection from circles and 
  * radiuses
  */
bool doCirclesIntersect(Point p1, double radius_1, Point p2, double radius_2);

/**
  * Create circles from hough space
  *
  * @circles[]   array of circles to be returned
  * @hough_space hough space of the image
  * @thr         threshold for considering a circle
  * @radius      maximum radius of a circle
  */
int createCircles(Circle circles[], Mat hough_space, double thr, double radius);

/**
  * Get average value in the hough space for a square
  *
  * @hough_space hoguh space for the image
  * @i,j         coordinates for the hough space
  * @height      height of the square
  * @width       width of the square
  * @return      the average value
  */
double getAverage(Mat hough_space, double i, double j, double width, double height);
/**
  * Combine list of squares together from the values in hough space
  *
  * @hough_space
  * @merged_squares[]  array of merged squares
  * @count 			   number of merged squares
  * @dartboards_vj     squares to be merged
  * @return 		   length of merged squares
  */
int combineSquares(Mat hough_space, square merged_squares[], int count, vector<Rect> dartboards_vj);

/**
  * Merge squares that resulted from hough transform
  * and from Viola Jones
  *
  * @merged_squaresp[] list of merged squares
  * @dartboards_vj     list of squares from VJ
  * @squares[]         list of squares from hough space
  * @length            length of squares[]
  * @hough_space       the hough space of the image
  * @return 		   size of merged list
  */
int mergeSquares(square merged_squares[], vector<Rect> dartboards_vj, square squares[], int length, Mat hough_space);

/**
  * Merge circles resulted from the hough space
  *
  * @circles[] list of resulted circles
  * @length    length of the list
  * @return    length of merged list
  */
int mergeCircles(Circle circles[], int length);

/**
  * Initialize some random radiuses
  */
Mat1d initialiseRadius(int rows, int cols, double low, double high);


/**
  * Calculate the hough space for circles
  * 
  * @grad_magnitude   the thresholded magnitude gradient
  *                   of the original image
  * @grad_orientation the gradient orientation image
  * @thr              threshold for circles
  * @radius           set of random radiuses
  * @radius_length    maximum length of radius set
  */
Mat getHoughSpace(Mat grad_magnitude, Mat grad_orientation, double thr, Mat1d radius, int radius_length);

/**
  * Function to apply threshold on an image
  *
  * @return thresholded image
  */
Mat applyThreshold(Mat image, double threshold);

/**
  * Apply multiple blur on an image
  *
  * @image  to be blurred
  * @thr    no. of times
  * @return blurred image
  */
Mat applyMultipleBlur(Mat image, double thr);

/**
  * Apply blur on image
  *
  * @return blurred image
  */
Mat applyBlur(Mat image);

/**
  * Convert image to gray
  *
  * @return converted image
  */
Mat convertToGray(Mat image);

/**
  * Calculate the derivative in the 
  * x direction for an image
  *
  * @return the gradient in the x direction
  */
Mat getXGrad(Mat image);


/**
  * Calculate the derivative in the 
  * y direction for an image
  *
  * @return the gradient in the y direction
  */
Mat getYGrad(Mat image);

/**
  * Calculate the magnitude gradient from
  * 
  * @grad_x gradient in the x direction
  * @grad_y gradient in the y direction
  * @return the magnitude gradient
  */
Mat calcMagnitude(Mat grad_x, Mat grad_y);

/**
  * Calculate the orientation gradient
  *
  * @grad_x gradient in the x direction
  * @grad_y gradient in the y direction
  * @return the orientation gradient
  */
Mat calcOrientation(Mat grad_x, Mat grad_y);


/**
  * Initialize matrix with 0 - int
  */
Mat initMatInt(int rows, int cols);

/**
  * Initialize a matrix with 0 - doubles
  */
Mat initMatDouble(int rows, int cols);

/**
  * Get center from given coordinates
  */
Point centerPoint(int x1, int y1, int x2, int y2);

/**
  * Read ground truth coordinates from file
  */
Mat readInCoordinates();

/**
  * Calculate f1 score from ground truth
  *
  * @coordinates from ground truth
  * @dartboards  detected squares at the end
  * @frame       image to draw on
  * @imageIndex  no. of image to open
  * @length      of the list of squares
  * @return      the f1 score
  */
double f1Score(Mat coordinates, square dartboards[], Mat frame, int imageIndex, int length);

/**
  * Perform Viola-Jones on given image
  *
  * @blurred_image
  * @return list of detected squares
  */
vector<Rect> detectViolaJones(Mat blurred_image);

/**
  * Function to create squares from circles
  */
void createSquares(Circle circles[], square squares[], int length);

int main( int argc, char* argv[] )
{
  // Read in the arguments from the console
  String imageName = argv[1];

  // Read in images
  Mat image = imread(imageName, CV_LOAD_IMAGE_UNCHANGED);

  if( argc != 2 || !image.data )
  {
    printf("No image data \n");
    return -1;  
  }

  // Load the Strong Classifier in a structure called `Cascade'
  if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  //Apply multiple blur on the image
  Mat multiple_blurred_image(image.rows, image.cols, CV_64FC1);
  multiple_blurred_image = applyMultipleBlur(image, BLUR_THR);

  //Apply blur to the image
  Mat blurred_image = applyBlur(image);
  Mat gray_image(image.rows, image.cols, CV_64FC1);
  gray_image = convertToGray(blurred_image);
  
  //Derivatives of the image in x and y directions
  Mat grad_x(image.rows, image.cols, CV_64FC1);
  Mat grad_y(image.rows, image.cols, CV_64FC1); 
  grad_x = getXGrad(gray_image);
  grad_y = getYGrad(gray_image);

  //Gradient magnitude image and thresholded 
  Mat magnitude_image(image.rows, image.cols, CV_64FC1);
  magnitude_image = calcMagnitude(grad_x, grad_y);
  magnitude_image = applyThreshold(magnitude_image, MAGN_THR);

  //Gradient orientation image
  Mat gradient_orientation(image.rows, image.cols, CV_64FC1);
  gradient_orientation = calcOrientation(grad_x, grad_y);
  
  //Initialize the radius vector
  Mat1d radius_array;
  radius_array = initialiseRadius(1, RADIUS_LENGTH, RADIUS_LOW, RADIUS_HIGH);

  //Generate 2d hough space for circles
  Mat hough_space_2d(image.rows, image.cols, CV_64FC1); 
  hough_space_2d = getHoughSpace(magnitude_image, gradient_orientation, HOUGH_THR, radius_array, RADIUS_LENGTH);

  //Create circles from hough space
  Circle circles[1000];
  int length;
  length = createCircles(circles, hough_space_2d, CENTER_THR, RADIUS);
  length = mergeCircles(circles, length);

  //Create squares from circles
  square squaresFromCircles[1000];
  createSquares(circles, squaresFromCircles, length);
  
  //Detect, threshold and draw squares from Viola Jones
  std::vector<Rect> dartboards_vj;
  dartboards_vj = detectViolaJones(multiple_blurred_image);
  thrSquares(hough_space_2d, dartboards_vj, SQUARES_THR);

  //Merge squares
  square merged_squares[1000]; 
  int merged_squares_length = mergeSquares(merged_squares, dartboards_vj, squaresFromCircles, length, hough_space_2d);
  merged_squares_length = thrRect(hough_space_2d, merged_squares, SQUARES_THR, merged_squares_length);
  
  //Calculate the F1 score
  // Mat coordinates = readInCoordinates();
  // float f1_score = f1Score(coordinates, merged_squares, image, 4, merged_squares_length);
  
  //Draw squares on the image
  //drawSquaresFromArray(merged_squares, image, merged_squares_length);
  drawSquaresFromArray(squaresFromCircles, image, length);
  //drawSquaresFromVector(dartboards_vj, image);
  
  //Display and output section
  //displayOriginalImage(image);
  normalize(image, image, 0, 255, NORM_MINMAX, CV_8UC1);
  imwrite("detected.jpg", image);
  
  return 0;
}


/* FUNCTIONS IMPLEMENTATIONS */


void displayImage(Mat image){
  //create a window for displaying an image
  namedWindow("Display window", CV_WINDOW_AUTOSIZE);

  //normalize the image
  normalize(image, image, 0, 1, NORM_MINMAX, CV_64FC1);

  //show the image
  imshow("Display window", image); 

  //wait for a key press until returning from the program
  waitKey(0);

  //free memory occupied by image
  image.release();
}

/**
  * Displaying original images
  */
void displayOriginalImage(Mat image){
  //create a window for displaying an image
  namedWindow("Display window1", CV_WINDOW_AUTOSIZE);

  //show the image
  imshow("Display window1", image); 

  //wait for key press
  waitKey(0);

  //free memory occupied by image
  image.release();
}

/**
  * Draw rectangles on a given image
  *
  * @squares vector of rectangles
  * @image   image to draw the rectangles on
  */
void drawSquaresFromArray(square squares[], Mat image, int length){
	for(int i = 0; i < length; i++){
		rectangle(image, Point(squares[i].x, squares[i].y), Point(squares[i].x + squares[i].width, squares[i].y + squares[i].height), Scalar( 0, 255, 0 ), 2);
	}
}

void drawSquaresFromVector(vector<Rect> squares, Mat image){
	for(int i = 0; i < squares.size(); i++){
		rectangle(image, Point(squares[i].x, squares[i].y), Point(squares[i].x + squares[i].width, squares[i].y + squares[i].height), Scalar( 255, 0, 0 ), 2);
	}
}

/**
  * Draw squares from cicles center points
  *
  * @image         image to draw on
  * @hough_space   hough space for circles
  * @thr           threshold for selecting a center
  * @radius        for determining the size of the squares
  */
void drawSquaresFromCenter(Mat image, Mat hough_space, double thr, double radius){
	Mat abs_hough_space(image.rows, image.cols, CV_8U);
  	normalize(hough_space, abs_hough_space, 0, 255, NORM_MINMAX, CV_8U);

 	for(int i = 0; i < abs_hough_space.rows; i++){
    	for(int j = 0; j < abs_hough_space.cols; j++){
      		if(abs_hough_space.at<uchar>(i,j) > thr){
        		Point pt1(j + radius, i - radius);
        		Point pt2(j - radius, i + radius);
				rectangle(image, pt1, pt2, Scalar( 0, 255, 0 ), 2);
      		}
    	}
  	}
}

/**
  * Draw circles from cicles center points
  *
  * @image         image to draw on
  * @hough_space   hough space for circles
  * @thr           threshold for selecting a center
  * @radius        draw circles of size of this radius
  */
void displayCircles(Circle circles[], int length, Mat image){
	for(int i = 0; i < length; i++){
		circle(image, circles[i].center, circles[i].radius, Scalar(255,0,0), 2, 8, 0);
	}
}

/**
  * Remove the squares that have an average value
  * in the hough space less then a threshold
  *
  * @hough_   space hough space for circles
  * @detected vector of detected squares
  * @thr      threshold for the average value
  */
void thrSquares(Mat hough_space, vector<Rect>& detected, double thr){
	Mat abs_hough_space(hough_space.rows, hough_space.cols, CV_8U);
    normalize(hough_space, abs_hough_space, 0, 255, NORM_MINMAX, CV_8U);
    
    //Array to keep the average value in the hough space
    double averages[detected.size()];
    for(int i = 0; i < detected.size(); i++){
    	averages[i] = 0; 
    }

    double sum = 0;
    double count = 0;
    double av;
    double max = averages[0];
    for(int k = 0; k < detected.size(); k++){
	    for(int i = detected[k].x; i <= detected[k].x + detected[k].height; i++){
				for(int j = detected[k].y; j <= detected[k].y + detected[k].width; j++){
					sum   += abs_hough_space.at<uchar>(j,i);
					count += 1;
				}
    	}
		av = sum/count;
		averages[k] = av;
		if(av > max){
			max = av;
		}
		sum   = 0;
		count = 0;
	}

	//Delete squares with average value less than
	//the threshols
	for(int i = 0; i < detected.size()-1; i++){
		if(averages[i] < 0.3 * max){
			detected.erase(detected.begin() + i);
			for(int j = i + 1; j < detected.size(); j++){
				averages[j - 1] = averages[j];
			}
			i--;
		}
	}
}

int thrRect(Mat hough_space, square squares[], double thr, int length){
	Mat abs_hough_space(hough_space.rows, hough_space.cols, CV_8U);
    normalize(hough_space, abs_hough_space, 0, 255, NORM_MINMAX, CV_8U);

    double averages[length];
    for(int i = 0; i < length; i++){
    	averages[i] = 0; 
    }

    double sum = 0;
    double count = 0;
    double av;
    double max = averages[0];
    for(int k = 0; k < length; k++){
	    for(int i = squares[k].x; i <= squares[k].x + squares[k].height; i++){
				for(int j = squares[k].y; j <= squares[k].y + squares[k].width; j++){
					sum   += abs_hough_space.at<uchar>(j,i);
					count += 1;
				}
    	}
		av = sum/count;
		averages[k] = av;
		if(av > max){
			max = av;
		}
		sum   = 0;
		count = 0;
	}

	for(int i = 0; i < length; i++){
		if(averages[i] < 0.46 * max){
			for(int j = i + 1; j < length - 1; j++){
				squares[j - 1]  = squares[j];
				averages[j - 1] = averages[j];
			}
			i--;
			length--;
		}
	}

	return length;
}

/**
  * Do intersection between two rectangles
  *
  * @return true or false
  */
bool doRectIntersect(Rect square_1, Rect square_2)
{
	double r1_x1 = square_1.x;
	double r1_y1 = square_1.y;

	double r1_x2 = square_1.x + square_1.height;
	double r1_y2 = square_1.y + square_1.width;

	double r2_x1 = square_2.x;
	double r2_y1 = square_2.y;

	double r2_x2 = square_2.x + square_2.height;
	double r2_y2 = square_2.y + square_2.width;

	bool noOverlap = 
		r1_x1 > r2_x2 ||
		r2_x1 > r1_x2 ||
		r1_y1 > r2_y2 ||
		r2_y1 > r1_y2;

	return !noOverlap;
}

/**
  * Do intersection between two squares
  *
  * @return true or false
  */
bool doSquareIntersect(square square_1, Rect square_2)
{
	double r1_x1 = square_1.x;
	double r1_y1 = square_1.y;

	double r1_x2 = square_1.x + square_1.height;
	double r1_y2 = square_1.y + square_1.width;

	double r2_x1 = square_2.x;
	double r2_y1 = square_2.y;

	double r2_x2 = square_2.x + square_2.height;
	double r2_y2 = square_2.y + square_2.width;

	bool noOverlap = 
		r1_x1 > r2_x2 ||
		r2_x1 > r1_x2 ||
		r1_y1 > r2_y2 ||
		r2_y1 > r1_y2;

	return !noOverlap;
}

/**
  * Calculate circle intersection from circles and 
  * radiuses
  */
bool doCirclesIntersect(Point p1, double radius_1, Point p2, double radius_2){
	double a = (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
	double b = (radius_1 + radius_2)*(radius_1 + radius_2);

	if (a <= b){
		return true;
	}
	return false;
}

/**
  * Create circles from hough space
  *
  * @circles[]   array of circles to be returned
  * @hough_space hough space of the image
  * @thr         threshold for considering a circle
  * @radius      maximum radius of a circle
  */
int createCircles(Circle circles[], Mat hough_space, double thr, double radius){
	Mat abs_hough_space(hough_space.rows, hough_space.cols, CV_8U);
  	normalize(hough_space, abs_hough_space, 0, 255, NORM_MINMAX, CV_8U);

  	//Go through values in hough space and keep the 
  	//ones above threshold
  	int count = 0;
	for(int i = 0; i < abs_hough_space.rows; i++){
		for(int j = 0; j < abs_hough_space.cols; j++){
			if(abs_hough_space.at<uchar>(i,j) > thr){
		 		Point center(j,i);
		 		circles[count].center = center;
				circles[count].radius = radius;
				circles[count].merged = false;
				count++;
			}
		}
	}
	return count;
}

/**
  * Get average value in the hough space for a square
  *
  * @hough_space hoguh space for the image
  * @i,j         coordinates for the hough space
  * @height      height of the square
  * @width       width of the square
  * @return      the average value
  */
double getAverage(Mat hough_space, double i, double j, double width, double height){	
	Mat abs_hough_space(hough_space.rows, hough_space.cols, CV_8U);
    normalize(hough_space, abs_hough_space, 0, 255, NORM_MINMAX, CV_8U);

	double av;
	double sum   = 0;
	double count = 0;
	for(double ii = i; ii < i + height; ii++){
		for(double jj = j; jj < j + width; jj++){
			sum   += abs_hough_space.at<uchar>(jj,ii);
			count += 1;
		}
	}
	av = sum/count;

	return av;
}

/**
  * Combine list of squares together from the values in hough space
  *
  * @hough_space
  * @merged_squares[]  array of merged squares
  * @count 			   number of merged squares
  * @dartboards_vj     squares to be merged
  * @return 		   length of merged squares
  */
int combineSquares(Mat hough_space, square merged_squares[], int count, vector<Rect> dartboards_vj){
	
	bool merged[dartboards_vj.size()];
	for(int i = 0; i < dartboards_vj.size(); i++){
		merged[i] = false;
	}


	for (int i = 0; i < dartboards_vj.size(); i++)
	{
		//Calculate intersect area
		int noSquares = 0;
		int sum_x = dartboards_vj[i].x;
		int sum_y = dartboards_vj[i].y;
		int sum_width  = dartboards_vj[i].width;
		int sum_height = dartboards_vj[i].height;
		double area_1  = dartboards_vj[i].width * dartboards_vj[i].height;
		for (int j = 0; j < dartboards_vj.size(); j++) 
		{
			if (i != j && merged[j] != true && merged[i] != true)
			{
				bool intersect = doRectIntersect(dartboards_vj[i], dartboards_vj[j]);
				double area_2 = dartboards_vj[j].width * dartboards_vj[j].height;
				double intersectionArea = max(0, min(dartboards_vj[j].x + dartboards_vj[j].width, (int)dartboards_vj[i].x + (int)dartboards_vj[i].width) 
					- max(dartboards_vj[j].x, (int)dartboards_vj[i].x))* max(0, min(dartboards_vj[j].y + dartboards_vj[j].height, (int)dartboards_vj[i].y + (int)dartboards_vj[i].height)
					- max(dartboards_vj[j].y, (int)dartboards_vj[i].y));
				double unionArea = area_1 + area_2 - intersectionArea;
				double ratio = intersectionArea / unionArea;
				
				//Check that they intersect more than a value
				if (intersect == true && ratio > 0.25)
				{
					noSquares++;
					sum_x = sum_x + dartboards_vj[j].x;
					sum_y = sum_y + dartboards_vj[j].y;
					sum_width= sum_width + dartboards_vj[j].width;
					sum_height= sum_height + dartboards_vj[j].height;
					merged[j]= true;
				}
			}
		}
		if (noSquares != 0)
		{
			dartboards_vj[i].x = sum_x/(noSquares + 1);
			dartboards_vj[i].y = sum_y/(noSquares + 1);

			dartboards_vj[i].width = sum_width/(noSquares + 1);
			dartboards_vj[i].height = sum_height/(noSquares + 1);

		} 
	}

	for(int i = 0; i < dartboards_vj.size(); i++){
		if(merged[i] != true){
			merged_squares[count].x = dartboards_vj[i].x;
			merged_squares[count].y = dartboards_vj[i].y;
			merged_squares[count].width = dartboards_vj[i].width;
			merged_squares[count].height = dartboards_vj[i].height;
			count++;
		}

	}

	return count;
	
}

/**
  * Merge squares that resulted from hough transform
  * and from Viola Jones
  *
  * @merged_squaresp[] list of merged squares
  * @dartboards_vj     list of squares from VJ
  * @squares[]         list of squares from hough space
  * @length            length of squares[]
  * @hough_space       the hough space of the image
  * @return 		   size of merged list
  */
int mergeSquares(square merged_squares[], vector<Rect> dartboards_vj, square squares[], int length, Mat hough_space){
	int count = 0;
	
	bool merged[dartboards_vj.size()];
	for(int i = 0; i < dartboards_vj.size(); i++){
		merged[i] = false;
	}

	//Go through all squares
	for(int i = 0; i < length; i++){

		double max_av = getAverage(hough_space, squares[i].x, squares[i].y, squares[i].width, squares[i].height);
		double index  = -1;
		double area_1  = squares[i].width * squares[i].height;

		for(int j = 0; j < dartboards_vj.size(); j++){

			//If they intersect, calculate intersect ratio and average value in hough space
			if(doSquareIntersect(squares[i], dartboards_vj[j]) && merged[j] == false){

				double av = getAverage(hough_space, dartboards_vj[j].x, dartboards_vj[j].y, dartboards_vj[j].width, dartboards_vj[j].height);
				double area_2 = dartboards_vj[j].width * dartboards_vj[j].height;
				double intersectionArea = max(0, min(dartboards_vj[j].x + dartboards_vj[j].width, (int)squares[i].x + (int)squares[i].width) 
					- max(dartboards_vj[j].x, (int)squares[i].x))* max(0, min(dartboards_vj[j].y + dartboards_vj[j].height, (int)squares[i].y + (int)squares[i].height)
					- max(dartboards_vj[j].y, (int)squares[i].y));
				double unionArea = area_1 + area_2 - intersectionArea;
				double ratio = intersectionArea / unionArea;
				if(av > max_av && ratio > 0.01){
					max_av = av;
					index = j;
				}
				merged[j] = true;
			}
		}
		if(index==-1){
			merged_squares[count] = squares[i];
		}else{
			merged_squares[count].x      = dartboards_vj[index].x;
			merged_squares[count].y      = dartboards_vj[index].y;
			merged_squares[count].width  = dartboards_vj[index].width;
			merged_squares[count].height = dartboards_vj[index].height;
		}
		count++;
	}

	//Remove squares that were merged
	int merged_size = dartboards_vj.size();
	for (int i = 0; i < dartboards_vj.size(); i++)
	{
		if(merged[i]== true)
		{   
			for(int j = i + 1; j < merged_size; j++){
				merged[j - 1] = merged[j];
			}
			merged_size--;
			dartboards_vj.erase(dartboards_vj.begin() + i);
			i--;
		}
	}

	count = combineSquares(hough_space, merged_squares, count, dartboards_vj);

	return count;
}

/**
  * Merge circles resulted from the hough space
  *
  * @circles[] list of resulted circles
  * @length    length of the list
  * @return    length of merged list
  */
int mergeCircles(Circle circles[], int length) 
{
	//Go through all circles and check if they intersect
	for (int i = 0; i < length; i++)
	{
		int noCircles = 0;
		int sum_x = circles[i].center.x;
		int sum_y = circles[i].center.y;
		int sum_radius = circles[i].radius;
		for (int j = 0; j < length; j++) 
		{
			if (i != j && circles[j].merged != true && circles[i].merged != true)
			{
				bool intersect = doCirclesIntersect(circles[i].center, circles[i].radius, circles[j].center, circles[j].radius);
				if (intersect == true)
				{
					noCircles++;
					sum_x = sum_x + circles[j].center.x;
					sum_y = sum_y + circles[j].center.y;
					sum_radius = sum_radius + circles[j].radius;
					circles[j].merged = true;
				}
			}
		}
		if (noCircles != 0)
		{
			circles[i].center = Point(sum_x / (noCircles + 1), sum_y / (noCircles + 1));
			circles[i].radius = sum_radius / (noCircles + 1);
		} 
	}

	//Remove circles that were merged
	for (int i = 0; i < length; i++)
	{
		if(circles[i].merged == true)
		{   
			for (int j = i + 1; j < length; j++)
			{
				circles[j - 1] = circles[j];
			}
			i--;
			length--;
		}
	}

	return length;
}

/**
  * Initialize some random radiuses
  */
Mat1d initialiseRadius(int rows, int cols, double low, double high){
  Mat1d radius(rows, cols);
  randu(radius, Scalar(low), Scalar(high));

  return radius;
}

/**
  * Calculate the hough space for circles
  * 
  * @grad_magnitude   the thresholded magnitude gradient
  *                   of the original image
  * @grad_orientation the gradient orientation image
  * @thr              threshold for circles
  * @radius           set of random radiuses
  * @radius_length    maximum length of radius set
  */
Mat getHoughSpace(Mat grad_magnitude, Mat grad_orientation, double thr, Mat1d radius, int radius_length){
  const int ROWS=grad_magnitude.rows; //+ 2 * max; 
  const int COLS=grad_magnitude.cols; //+ 2 * max; 
  int dims[3] = {ROWS, COLS, radius_length};
  cv::Mat hough_space = cv::Mat(3, dims, CV_64FC1);

  for(int i = 0; i < grad_magnitude.rows; i++){
    for(int j = 0; j < grad_magnitude.cols; j++){
      for(int k = 0; k < radius.cols; k++){
        if(grad_magnitude.at<double>(i,j) > thr){
          int i0 = i + radius.at<double>(0,k) * cos(grad_orientation.at<double>(i,j));
          int j0 = j + radius.at<double>(0,k) * sin(grad_orientation.at<double>(i,j));
          int i1 = i - radius.at<double>(0,k) * cos(grad_orientation.at<double>(i,j));
          int j1 = j - radius.at<double>(0,k) * sin(grad_orientation.at<double>(i,j));

          if(i0 < grad_magnitude.rows && j0 < grad_magnitude.cols && i0 >= 0 && j0 >= 0){
            hough_space.at<double>(i0,j0,k) += 1;
          }
          if(i1 < grad_magnitude.rows && j1 < grad_magnitude.cols && i1 >= 0 && j1 >= 0){
            hough_space.at<double>(i1,j1,k) += 1;
          }
        }
      }
    }
  }

  //Convert hough space from 3D to 2D
  Mat hough_space_2d(grad_magnitude.rows, grad_magnitude.cols, CV_64FC1);
  double sum = 0;
  for(int i = 0; i < hough_space_2d.rows; i++){
    for(int j = 0; j < hough_space_2d.cols; j++){
      for(int k = 0; k < radius.cols; k++){
        sum += hough_space.at<double>(i,j,k);
      }
      hough_space_2d.at<double>(i,j) = sum;
      sum = 0;
    }
  }

  return hough_space_2d;
}

/**
  * Function to apply threshold on an image
  *
  * @return thresholded image
  */
Mat applyThreshold(Mat image, double threshold){
  Mat abs_image(image.rows, image.cols, CV_64FC1);

  normalize(image, abs_image, 0, 255, NORM_MINMAX, CV_64FC1);
  
  for(int i = 0; i < abs_image.rows; i++){
     for(int j = 0; j < abs_image.cols; j++){
       if(abs_image.at<double>(i,j) > threshold){
          abs_image.at<double>(i,j) = 255;
       } else {
          abs_image.at<double>(i,j) = 0;
       }
    }
  }
  return abs_image;
}

/**
  * Apply multiple blur on an image
  *
  * @image  to be blurred
  * @thr    no. of times
  * @return blurred image
  */
Mat applyMultipleBlur(Mat image, double thr){
	Mat blurred_image;
	GaussianBlur(image, blurred_image, Size(3,3), 0, 0, BORDER_DEFAULT);
	for(int i = 0; i < BLUR_THR; i++){
		GaussianBlur( blurred_image, blurred_image, Size(3,3), 0, 0, BORDER_DEFAULT );
	}
	return blurred_image;
}

/**
  * Apply blur on image
  *
  * @return blurred image
  */
Mat applyBlur(Mat image){
  Mat blurred_image;
  GaussianBlur( image, blurred_image, Size(3,3), 0, 0, BORDER_DEFAULT );
  return blurred_image;
}

/**
  * Convert image to gray
  *
  * @return converted image
  */
Mat convertToGray(Mat image){
  cvtColor(image, image, CV_RGB2GRAY);
  
  return image;
}

/**
  * Calculate the derivative in the 
  * x direction for an image
  *
  * @return the gradient in the x direction
  */
Mat getXGrad(Mat image){
  int scale = 1;
  int delta = 0;
  int ddepth = CV_64FC1;

  Mat grad_x;
  
  // Apply the Sobel edge detector
  Sobel( image, grad_x, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

  return grad_x;
}

/**
  * Calculate the derivative in the 
  * y direction for an image
  *
  * @return the gradient in the y direction
  */
Mat getYGrad(Mat image){
  int scale = 1;
  int delta = 0;
  int ddepth = CV_64FC1;

  Mat grad_y;
  
  //Apply the Sobel edge detector
  Sobel( image, grad_y, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );

  return grad_y;  
}

/**
  * Calculate the magnitude gradient from
  * 
  * @grad_x gradient in the x direction
  * @grad_y gradient in the y direction
  * @return the magnitude gradient
  */
Mat calcMagnitude(Mat grad_x, Mat grad_y){
  Mat grad, abs_grad_x, abs_grad_y;

  convertScaleAbs( grad_x, abs_grad_x );
  convertScaleAbs( grad_y, abs_grad_y );

  // Finally, we try to approximate the gradient by adding both directional gradients
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

  return grad;
}

/**
  * Calculate the orientation gradient
  *
  * @grad_x gradient in the x direction
  * @grad_y gradient in the y direction
  * @return the orientation gradient
  */
Mat calcOrientation(Mat grad_x, Mat grad_y){
  Mat grad_orientation(grad_x.rows, grad_x.cols, CV_64FC1);
  phase(grad_x, grad_y, grad_orientation);
  return grad_orientation;
}

/**
  * Initialize matrix with 0 - int
  */
Mat initMatInt(int rows, int cols) {
	Mat matrice(rows, cols, CV_32S);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			matrice.at<int>(i, j) = 0;
		}
	}
	return matrice;
}

/**
  * Initialize a matrix with 0 - doubles
  */
Mat initMatDouble(int rows, int cols){
	Mat matrice(rows, cols, CV_64FC1);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			matrice.at<double>(i, j) = 0.0;
		}
	}
	return matrice;
}

/**
  * Get center from given coordinates
  */
Point centerPoint(int x1, int y1, int x2, int y2)
{
	int height = abs(x1 - x2)/2;
	int width = abs(y1 - y2) / 2;
	Point center = Point(x2 - height, y2 - width);

	return center;
}

/**
  * Read ground truth coordinates from file
  */
Mat readInCoordinates() {
	Mat coordinates = initMatInt(16,13);
	for(int i = 0; i < 16; i++){
		for(int j = 0; j < 13; j++){
			coordinates.at<int>(i,j) = 0;
		}
	}

	ifstream myfile;
	myfile.open("coordinates.txt", ios::in);
	if (myfile.is_open()){
		for (int i = 0; i < 16; i++)
		{

			int count;
			myfile >> i;
			myfile >> count;
			coordinates.at<int>(i, 0) = count;
			for (int j = 1; j < (count * 4 + 1); j++)
			{	
				myfile >> coordinates.at<int>(i, j);
			}
		}

	} else {
		printf("Error opening the coordinates file\n");
	}

	myfile.close();

	return coordinates;
}

/**
  * Calculate f1 score from ground truth
  *
  * @coordinates from ground truth
  * @dartboards  detected squares at the end
  * @frame       image to draw on
  * @imageIndex  no. of image to open
  * @length      of the list of squares
  * @return      the f1 score
  */
double f1Score(Mat coordinates, square dartboards[], Mat frame, int imageIndex, int length)
{
	int count = coordinates.at<int>(imageIndex, 0);

	double truePositives = count;
	double correctPositives = 0;
	int returnedPositives = length ;

	Mat maxRatios = initMatInt(1, count);
	int noRect = 0;

	//Go through all the squars in the file
	for (int j = 1; j < (count * 4 + 1); j = j + 4){
		int x1 = coordinates.at<int>(imageIndex, j);
		int y1 = coordinates.at<int>(imageIndex, j + 1);
		int x2 = coordinates.at<int>(imageIndex, j + 2);
		int y2 = coordinates.at<int>(imageIndex, j + 3);

		Point upLeftCorner(x1, y1);
		Point downRightCorner(x2, y2);
		Point center = centerPoint(x1, y1, x2, y2);
		rectangle((frame), upLeftCorner, downRightCorner, Scalar(0, 0, 255));

		//Calculate area of intersection
		double area1 = abs(x1 - x2) * abs(y1 - y2);
		for (int i = 0; i < length; i++){
			double area2 = dartboards[i].width * dartboards[i].height;
			double intersectionArea = max(0, min((int)dartboards[i].x + (int)dartboards[i].width, x2) - max((int)dartboards[i].x, x1))* max(0, min((int)dartboards[i].y + (int)dartboards[i].height, y2) - max((int)dartboards[i].y, y1));
			double unionArea = area1 + area2 - intersectionArea;
			double ratio = intersectionArea / unionArea;
			if (center.x > dartboards[i].x && center.x < (dartboards[i].x + dartboards[i].width) && center.y > dartboards[i].y && center.y < (dartboards[i].y + dartboards[i].y + dartboards[i].height)){
				maxRatios.at<int>(1, noRect) = 1;
			}
		}
		noRect++;
	}

	//Increase the no. of correctly detected
	//positives
	for (int i = 0; i < count; i++){
		if (maxRatios.at<int>(1, i) != 0){
			correctPositives++;
		}
	}

	//Compute the f1 score
	double precision = correctPositives / returnedPositives;
	double recall = correctPositives/ truePositives;
	double f1_score = 2 * (precision * recall) / (precision + recall);
	return f1_score;
}

/**
  * Perform Viola-Jones on given image
  *
  * @blurred_image
  * @return list of detected squares
  */
vector<Rect> detectViolaJones(Mat blurred_image){
	std::vector<Rect> dartboards_vj;
	Mat image_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( blurred_image, image_gray, CV_BGR2GRAY ); //should be blurred_image(doing part2 now)
	equalizeHist( image_gray, image_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( image_gray, dartboards_vj, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

    // 3. Print number of dartboards found
	//std::cout << dartboards_vj.size() << std::endl;

	return dartboards_vj; 
}

/**
  * Function to create squares from circles
  */
void createSquares(Circle circles[], square squares[], int length)
{

	for(int i = 0; i < length; i++){
		double diameter = 2 * circles[i].radius;
		squares[i].x      = circles[i].center.x - circles[i].radius;
		squares[i].y      = circles[i].center.y - circles[i].radius;
		squares[i].width  = diameter;
		squares[i].height = diameter;
	}
}