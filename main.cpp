#include <iostream>
#include "yolo.hpp"
#include "sort_tracking.h"

#define PATH_VIDEO "/home/minh/Documents/test_tracking/data_test/testaa.mkv"
#define PATH_MODEL "/home/minh/Documents/test_tracking/models/yolov5s.engine"

using namespace WrappingYolo;

int main()
{
	cv::VideoCapture cap(PATH_VIDEO);
        int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	if(!cap.isOpened())
	{
		std::cout << "Cannot open video!\n";
		return -1;
	}

	Infer detecter(PATH_MODEL);
	cv::VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(frame_width,frame_height));

	SortTracking tracker("Abcdef"); 	
	int count = 0;
	while(1)
	{
		cv::Mat frame, outDetect;
		cap >> frame;
		if(frame.empty())
		{
			std::cout << "Empty\n";
			break;
		}
		count ++;
		//if(count%10 != 0)
		//{
			//continue;
		//}		

		BoxArray box_array;
		detecter.doInference(frame,box_array ,outDetect);

		std::vector<TrackingBox> inputs;
		for(int i = 0 ; i < box_array.size(); i ++)
		{
			TrackingBox input;
			input.box.x = box_array[i].left;
			input.box.y = box_array[i].top;
			input.box.width = box_array[i].right - box_array[i].left;
			input.box.height = box_array[i].bottom - box_array[i].top;
			inputs.push_back(input); 
		}

		std::vector<TrackingBox> result;

		tracker.update(inputs, result);

		for(int i = 0 ; i < result.size(); i ++)
		{
			cv::rectangle(frame, result[i].box, cv::Scalar(0x27, 0xC1, 0x36), 1);          
       		cv::putText(frame, std::to_string(result[i].id), cv::Point(result[i].box.x, result[i].box.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    	}

    	//cv::imshow("Show", frame);
    	video.write(frame);
    	if(cv::waitKey(1) == 27)
    		break;        
	}
}
