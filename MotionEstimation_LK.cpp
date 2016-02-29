////////////////////////**** EEE 508 Project #2  ****///////////////////////
//
//	File Name:	 MotionEstimation_LK.cpp
//	Author:		 Ramsundar K G
//	Date:		 28 April 2015
//
//	Description: This program performs motion estimation using
//				 Lucas-Kanade method.
//				 Some parts of the code are adapted from lkdemo.cpp 
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
#define SURF_HESSIAN_THRESH	400

#define SOURCE_FILE			1
#define SOURCE_CAM			2
#define FEATURE_TYPE_SIFT	1
#define FEATURE_TYPE_SURF	2

#define RES_X	320
#define RES_Y	280

#define FEATURE_POINTS_WINDOW		"Feature Point Detection:"
#define OPTICAL_FLOW_WINDOW			"Optical Flow of Feature Points:"
#define OPTICAL_FLOW_OVER_WINDOW	"Optical Flow Overlaid on Original Image:"

#define WRITE_FILE_OPT_FLOW		0
/********************************************************************/


int Video_Source = SOURCE_FILE;
int FeatureDetector_Type = FEATURE_TYPE_SIFT;
int FeatureDetector_Num = FEAT_KEYPOINTS_MAX;


/********************************************************************/
// Function:	Print_Title()
// Description:	Prints Title in console window
/********************************************************************/
void Print_Title()
{
	cout<<"==================================================\n";
	cout<<"                 Project: #2                      \n";
	cout<<"      Feature Detectors | Motion Estimation       \n";
	cout<<"      ** Optical Flow using Lucas-Kanade **       \n";
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

	cout<<"\nEnter the Type of Feature Detector for First Frame ";
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
	
	Mat image_gray, image_gray_prev, image_color, image_flow;
    vector<Point2f> FeaturePoints_prev;
    vector<Point2f> FeaturePoints_new;
    vector<Scalar> FeaturePoints_Color;
	vector<bool> FeaturePoints_Status;
	vector<KeyPoint> keypoints;
	FeatureDetector* Detector;

	VideoCapture cap;
	
    TermCriteria LK_termcrit( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
    Size LK_winSize(31,31);

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
	
	#if WRITE_FILE_OPT_FLOW
		ofstream File_Out;
		if(FeatureDetector_Type == FEATURE_TYPE_SIFT)
			File_Out.open ( "OpticalFlow_Out_LK_SIFT.txt" );
		else if(FeatureDetector_Type == FEATURE_TYPE_SURF)
			File_Out.open ( "OpticalFlow_Out_LK_SURF.txt" );
	#endif

	// Create image windows
    namedWindow( FEATURE_POINTS_WINDOW, 1 );
	cvMoveWindow( FEATURE_POINTS_WINDOW, 50, 50);
	namedWindow( OPTICAL_FLOW_WINDOW, 1 );
	cvMoveWindow( OPTICAL_FLOW_WINDOW, 400, 50);
	namedWindow( OPTICAL_FLOW_OVER_WINDOW, 1);
	cvMoveWindow( OPTICAL_FLOW_OVER_WINDOW, 750, 50);
	
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
			image_flow_init = false;
		}

        if( Init_FeaturePoints )	// Init Feature Points for first frame
		{	
			// Define detector & extractor -- SIFT/SURF 
			if(FeatureDetector_Type == FEATURE_TYPE_SIFT)
			{
				Detector = new SiftFeatureDetector( FeatureDetector_Num );
			}
			else if(FeatureDetector_Type == FEATURE_TYPE_SURF)
			{
				Detector = new SurfFeatureDetector( SURF_HESSIAN_THRESH );
			}

			// Detect feature points
			Detector->detect(image_gray, keypoints);
			
			KeyPoint::convert(keypoints, FeaturePoints_new);
			FeaturePoints_new.resize( FeatureDetector_Num );
			FeaturePoints_Status.resize( FeaturePoints_new.size() );
			FeaturePoints_Color.resize( FeaturePoints_new.size() );
			
			#if WRITE_FILE_OPT_FLOW
				File_Out.close();
				File_Out.open( getFileName() );
				File_Out<<"Frame#\t";
			#endif
			
			for(size_t i=0; i < FeaturePoints_new.size(); i++)
			{
				FeaturePoints_Color[i] = RandColor();
				FeaturePoints_Status[i] = true;
				#if WRITE_FILE_OPT_FLOW
					File_Out<< "Feat_Pt_"<< i << "\t";
				#endif
			}
			#if WRITE_FILE_OPT_FLOW
				File_Out<<"\n";
			#endif
			
			image_flow = Mat(frame.size(), frame.type(), Scalar(0, 0, 0));
			Init_FeaturePoints = false;
        }
        else if( !FeaturePoints_prev.empty() )
        {
            vector<uchar> LK_status;
            vector<float> LK_err;
			
            if(image_gray_prev.empty())
                image_gray.copyTo(image_gray_prev);
			
			// Calculate the Opical Flow of the detected Keypoints using Pyramidal Lucas-Kanade method
            calcOpticalFlowPyrLK( image_gray_prev, image_gray, FeaturePoints_prev, FeaturePoints_new, LK_status, LK_err, LK_winSize,
                                  3, LK_termcrit, 0, 0.001);
            
			#if WRITE_FILE_OPT_FLOW
				File_Out<< frame_num <<"\t";
			#endif

			size_t j = 0;
			size_t k = 0;
            for(size_t i=0; i < FeaturePoints_Status.size(); i++)
            {
				if(FeaturePoints_Status[i] == false)
				{
					#if WRITE_FILE_OPT_FLOW
						File_Out<< " " <<"\t";
					#endif
					continue;
				}

				if( !LK_status[j] )
				{
					#if WRITE_FILE_OPT_FLOW
						File_Out<< " " <<"\t";
					#endif
					FeaturePoints_Status[i] = false;
					j++;
					continue;
				}
				
				FeaturePoints_Status[i] = true;

				// Draw Optical Flow of points
				line(image_flow, FeaturePoints_new[j], FeaturePoints_prev[j], FeaturePoints_Color[j], 2);
	
				FeaturePoints_new[k] = FeaturePoints_new[j];
				FeaturePoints_Color[k] = FeaturePoints_Color[k];
				// Highlight Feature points using circle
				circle( image_color, FeaturePoints_new[j], 3, FeaturePoints_Color[j], -1, 8);
				
				#if WRITE_FILE_OPT_FLOW
					File_Out<< (Point)FeaturePoints_new[k] <<"\t";
				#endif

				j++;
				k++;
			}
            FeaturePoints_new.resize(k);
			FeaturePoints_Color.resize(k);
			
			#if WRITE_FILE_OPT_FLOW
				File_Out<<"\n";
			#endif
        }

		// Display the results
        imshow( FEATURE_POINTS_WINDOW, image_color );
		imshow( OPTICAL_FLOW_WINDOW, image_flow);
		add(image_flow, image_color, image_color);
		imshow( OPTICAL_FLOW_OVER_WINDOW , image_color);
		
        char c = (char)waitKey(1);
        if( c == 27 )
            break;
		else if(c == 'i')
			Init_FeaturePoints = true;
		else if(c == 'c')
			FeaturePoints_new.clear(), FeaturePoints_prev.clear(), FeaturePoints_Status.clear();

        swap(FeaturePoints_new, FeaturePoints_prev);
        swap(image_gray_prev, image_gray);
    }
	
	// Execution time Calc --> Stop Time
	auto Time_Stop = getTickCount();
	// Execution time Calc --> Time elapsed
	long long Time_Elapsed = ((Time_Stop - Time_Start)*1000)/getTickFrequency();
	long double Time_PerFrame = Time_Elapsed / frame_num;

	destroyWindow( FEATURE_POINTS_WINDOW );
	destroyWindow( OPTICAL_FLOW_WINDOW );
	destroyWindow( OPTICAL_FLOW_OVER_WINDOW );

	#if WRITE_FILE_OPT_FLOW
		File_Out.close();
	#endif
	
	cout<<"\n\nTotal Time Elapsed: "<<Time_Elapsed<<" ms";
	cout<<"\nNumber of Frames Processed: "<<frame_num;
	cout<<"\nTime Taken (Avg) for Single frame: "<<Time_PerFrame<<" ms/frame";

	cout<<"\n\nPress Any Key to Exit...";
	getch();
    return 0;
}
