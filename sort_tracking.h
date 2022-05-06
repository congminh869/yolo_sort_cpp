#ifndef SORT_TRACKING_H
#define SORT_TRACKING_H

#include <iostream>
#include <cstring>
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>
#include <dirent.h>
#include <vector>
#include <set>
#include <assert.h>
#include <ctime>
#include <opencv2/core.hpp>
#include "Hungarian.h"
#include "KalmanTracker.h"

typedef struct TrackingBox
{
	//int frame;
	int id;
	Rect_<float> box;
}TrackingBox;

class SortTracking
{
public:
	SortTracking(std::string _monitor_id);
	~SortTracking();

	int update(std::vector<TrackingBox> input, std::vector<TrackingBox> &result);
	void reset()
	{
		trackers.clear();
	}

	std::string monitor_id; 

private:
	unsigned int frame_count = 0;
	int max_age = 10;
	int min_hits = 3;
	double iouThreshold = 0.05;
	std::vector<KalmanTracker> trackers;

	unsigned int trkNum = 0;
	unsigned int detNum = 0;

	int total_id = 0;
}; 

#endif
