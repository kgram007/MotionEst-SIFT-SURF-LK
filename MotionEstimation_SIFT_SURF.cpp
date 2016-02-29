////////////////////////**** EEE 508 Project #2  ****///////////////////////
//
//	File Name:	 MotionEstimation_SIFT_SURF.cpp
//	Author:		 Ramsundar K G
//	Date:		 28 April 2015
//
//	Description: This program performs motion estimation using
//				 SIFT and SURF feature detectors
//
//////////////////////////////////////////////////////////////////////////

/*************************  Includes  *******************************/
#include <stdio.h>
#include <conio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "cv.h"
#include "highgui.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
/********************************************************************/

using namespace cv;
using namespace std;

/**************************  Macros  ********************************/
#define INPUT_VIDEO_FILE	"Car.mp4"
#define CAMERA_INDEX		1

#define FEAT_KEYPOINTS_MAX	40
#define MAX_DISTANCE_RATIO	(0.8)
#define SURF_HESSIAN_THRESH	400

#define SOURCE_FILE			1
#define SOURCE_CAM			2
#define FEATURE_TYPE_SIFT	1
#define FEATURE_TYPE_SURF	2
#define OPT_FLOW_TYPE_LK	1
#define OPT_FLOW_TYPE_KLT	2

#define RES_X	320
#define RES_Y	280

#define FEATURE_POINTS_WINDOW		"Feature Point Detection:"
#define OPTICAL_FLOW_WINDOW			"Optical Flow of Feature Points:"
#define OPTICAL_FLOW_OVER_WINDOW	"Optical Flow Overlaid on Original Image:"
#define FEATURE_POINTS_MATCH_WINDOW	"Feature Point Matching:"
/********************************************************************/


int Video_Source = SOURCE_FILE;
int FeatureDetector_Type = FEATURE_TYPE_SURF;
int OpticalFlow_Type = OPT_FLOW_TYPE_LK;
int FeatureDetector_Num = FEAT_KEYPOINTS_MAX;


/********************************************************************/
// Function:	Print_Title()
// Description:	Prints Title in console window
/********************************************************************/
void Print_Title()
{
	cout<<"==================================================\n";
	cout<<"                   Project: #2                    \n";
	cout<<"     Feature Detectors | Motion Estimation        \n";
	cout<<"     ** Optical Flow using SIFT and SURF **	     \n";
	cout<<"                                                  \n";
	cout<<"==================================================\n";
}


/********************************************************************/
// Function:	getInputs()
// Description:	Get input from console window
/********************************************************************/
void getInputs()
{
	cout<<"\nEnter the Video Source ";
	cout<<"(1.Video File | 2.Camera): ";
	cin>> Video_Source;
	if( Video_Source != SOURCE_FILE &&
		Video_Source != SOURCE_CAM )
	{
		cout<<"Error: Invalid Source Option !!\n";
		getch();
		exit(EXIT_FAILURE);
	}

	cout<<"\nEnter the Type of Feature Detector ";
	cout<<"(1.SIFT | 2.SURF): ";
	cin>> FeatureDetector_Type;
	if( FeatureDetector_Type != FEATURE_TYPE_SIFT &&
		FeatureDetector_Type != FEATURE_TYPE_SURF )
	{
		cout<<"Error: Invalid Feature Detector Type !!\n";
		getch();
		exit(EXIT_FAILURE);
	}
}


/********************************************************************/
// Function:	RandColor()
// Description:	Generate Random RGB Color
/********************************************************************/
Scalar RandColor()
{
	unsigned char R = rand() % 256;
	unsigned char G = rand() % 256;
	unsigned char B = rand() % 256;

	return Scalar(R, G, B);
}


/********************************************************************/
// Function:	main()
// Description:	Program Starts Here!!
/********************************************************************/
int main()
{
	int frame_num = 0;
	bool image_flow_init = true;
	bool Init_FeaturePoints = true;
	
	Mat image_gray, image_gray_prev, image_color, image_match, image_flow;
	vector<Point2f> FeaturePoints_new;
	vector<Point2f> FeaturePoints_prev;
	vector<Scalar> FeaturePoints_Color;
	vector<bool> FeaturePoints_Status;
	
	vector<KeyPoint> Keypoints_new;
	vector<KeyPoint> Keypoints_prev;

	Mat FeatureDescriptor_new;
	Mat FeatureDescriptor_prev;

	FeatureDetector* Detector;
	DescriptorExtractor* Extractor;
	
	VideoCapture cap;
	
	//////////////////
	srand(time(NULL));
	Print_Title();
	getInputs();
	
	// Select video source
	if( Video_Source == SOURCE_FILE )
		cap = VideoCapture( INPUT_VIDEO_FILE );
	else if( Video_Source == SOURCE_CAM )
		cap.open( CAMERA_INDEX );

	// Check for proper init of source
	if( !cap.isOpened() )
	{
		cout << "\nError Opening Source !!\n";
		getch();
		exit(EXIT_FAILURE);
	}
	
	// Create image windows
	namedWindow( FEATURE_POINTS_WINDOW, 1 );
	cvMoveWindow( FEATURE_POINTS_WINDOW, 50, 50);
	namedWindow( OPTICAL_FLOW_WINDOW, 1 );
	cvMoveWindow( OPTICAL_FLOW_WINDOW, 400, 50);
	namedWindow( OPTICAL_FLOW_OVER_WINDOW, 1);
	cvMoveWindow( OPTICAL_FLOW_OVER_WINDOW, 750, 50);
	namedWindow( FEATURE_POINTS_MATCH_WINDOW, 1);
	cvMoveWindow( FEATURE_POINTS_MATCH_WINDOW, 150, 400);

	// Execution time Calc --> Start Time
	auto Time_Start = getTickCount();
	while(1)
	{
		Mat frame;
		cap >> frame;
		frame_num++;
		if( frame.empty() )
			break;

		resize(frame, frame, Size(RES_X, RES_Y));
		frame.copyTo(image_color);
		cvtColor(image_color, image_gray, COLOR_BGR2GRAY);
		
		// Init Optical Flow image
		if(image_flow_init == true)
		{
			image_flow = Mat(frame.size(), frame.type(), Scalar(0, 0, 0));
			image_match = Mat(frame.size(), frame.type(), Scalar(0, 0, 0));
			image_flow_init = false;
		}
		
		if( Init_FeaturePoints )	// Init Feature Points for first frame
		{	
			// Define detector & extractor -- SIFT/SURF 
			if(FeatureDetector_Type == FEATURE_TYPE_SIFT)
			{
				Detector = new SiftFeatureDetector( FeatureDetector_Num );
				Extractor = new SiftDescriptorExtractor();
			}
			else if(FeatureDetector_Type == FEATURE_TYPE_SURF)
			{
				Detector = new SurfFeatureDetector( SURF_HESSIAN_THRESH );
				Extractor = new SurfDescriptorExtractor();
			}

			// Detect feature points
			Detector->detect(image_gray, Keypoints_new);
			Extractor->compute(image_gray, Keypoints_new, FeatureDescriptor_new);
				
			Keypoints_new.resize( min(FeatureDetector_Num, (const int)Keypoints_new.size()) );
			Keypoints_prev = Keypoints_new;
			KeyPoint::convert( Keypoints_new, FeaturePoints_new );
			FeaturePoints_Status.resize( FeaturePoints_new.size() );
			FeaturePoints_Color.resize( FeaturePoints_new.size() );
			
			for(size_t i=0; i < FeaturePoints_new.size(); i++)
			{
				FeaturePoints_Color[i] = RandColor();
				FeaturePoints_Status[i] = true;
			}

			image_flow_init = true;
			Init_FeaturePoints = false;
		}
		else if( !FeaturePoints_prev.empty() )
		{			
			if( image_gray_prev.empty() )
				image_gray.copyTo(image_gray_prev);

			// Detect keypoints in each frame
			Detector->detect(image_gray, Keypoints_new);
			Keypoints_new.resize( min(FeatureDetector_Num, (const int)Keypoints_new.size()) );
			Extractor->compute(image_gray, Keypoints_new, FeatureDescriptor_new);
			Extractor->compute(image_gray_prev, Keypoints_prev, FeatureDescriptor_prev);
			
			vector<vector<DMatch>> Desc_Matches;
			vector<DMatch> Desc_Matches_Good;
			FlannBasedMatcher Desc_Matcher = FlannBasedMatcher();

			// Match feature points between frame 'n-1' and frame 'n' 
 			Desc_Matcher.knnMatch((const Mat)FeatureDescriptor_prev, (const Mat)FeatureDescriptor_new, Desc_Matches, 2);

			// Discard features points that are not good matches
			size_t k = 0;
			for(int i=0; i<min(FeatureDetector_Num, (const int)Keypoints_new.size()); i++)
			{
				float ratio;

				if(FeaturePoints_Status[i] == false)
				{
					FeaturePoints_Status[i] = true;
					continue;
				}
				FeaturePoints_Status[i] = false;

				if(Desc_Matches[k].size() == 1)
					ratio = 0;
				else if(Desc_Matches[k].size() == 2)
					ratio = (Desc_Matches[k][0].distance/Desc_Matches[k][1].distance);
				else
				{
					k++;
					continue;
				}

				if(ratio <= MAX_DISTANCE_RATIO)
				{
					Point a = Keypoints_new[ Desc_Matches[k][0].trainIdx ].pt;
					Point b = Keypoints_prev[k].pt;
					if(abs(a.x-b.x) < 5 && abs(a.y-b.y) < 5)
					{
						Desc_Matches_Good.push_back( Desc_Matches[k][0] );
						FeaturePoints_Status[i] = true;
						
						// Highlight the detected points and draw the optical flow
						circle( image_color, a, 3, FeaturePoints_Color[ k ], -1, 8);
						line(image_flow, a, b, FeaturePoints_Color[ k ], 2);
					}
				}
				k++;
			}
			
			KeyPoint::convert(Keypoints_new, FeaturePoints_new);
			KeyPoint::convert(Keypoints_prev, FeaturePoints_prev);

			// Draw the mapping of the features points in frame 'n-1' and frame 'n'
			drawMatches( image_gray_prev, Keypoints_prev, image_gray, Keypoints_new, Desc_Matches_Good, image_match);
			
			//Keypoints_new.resize( min(FeatureDetector_Num, (const int)Keypoints_new.size()) );
		}

		// Display the results
		imshow( FEATURE_POINTS_WINDOW, image_color );
		imshow( OPTICAL_FLOW_WINDOW, image_flow);
		add(image_flow, image_color, image_color);
		imshow( OPTICAL_FLOW_OVER_WINDOW , image_color);
		imshow( FEATURE_POINTS_MATCH_WINDOW , image_match);

		char c = (char)waitKey(1);
		if( c == 27 )
			break;
		else if(c == 'i')
			Init_FeaturePoints = true;
		else if(c == 'c')
			FeaturePoints_new.clear(), FeaturePoints_prev.clear(), FeaturePoints_Status.clear();

		swap(FeaturePoints_new, FeaturePoints_prev);
		swap(image_gray, image_gray_prev);
		swap(Keypoints_new, Keypoints_prev);
	}

	// Execution time Calc --> Stop Time
	auto Time_Stop = getTickCount();
	// Execution time Calc --> Time elapsed
	long long Time_Elapsed = ((Time_Stop - Time_Start)*1000)/getTickFrequency();
	long double Time_PerFrame = Time_Elapsed / frame_num;

	destroyWindow( FEATURE_POINTS_WINDOW );
	destroyWindow( OPTICAL_FLOW_WINDOW );
	destroyWindow( OPTICAL_FLOW_OVER_WINDOW );
	destroyWindow( FEATURE_POINTS_MATCH_WINDOW );
	
	cout<<"\n\nTotal Time Elapsed: "<<Time_Elapsed<<" ms";
	cout<<"\nNumber of Frames Processed: "<<frame_num;
	cout<<"\nTime Taken (Avg) for Single frame: "<<Time_PerFrame<<" ms/frame";

	cout<<"\n\nPress Any Key to Exit...";
	getch();

	return 0;
}
