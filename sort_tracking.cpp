#include "sort_tracking.h"

SortTracking::SortTracking(std::string _monitor_id)
{
	monitor_id =  _monitor_id;
}

SortTracking::~SortTracking()
{

}

double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}

int SortTracking::update(std::vector<TrackingBox> input, std::vector<TrackingBox> &result)
{
	KalmanTracker::kf_count = 0;

	std::vector<cv::Rect_<float>> predictedBoxes;
	std::vector<vector<double>> iouMatrix;
	std::vector<int> assignment;
	std::set<int> unmatchedDetections;
	std::set<int> unmatchedTrajectories;
	std::set<int> allItems;
	std::set<int> matchedItems;
	std::vector<cv::Point> matchedPairs;


	if (trackers.size() == 0) // the first frame met
	{
		// initialize kalman trackers using first detections.
		for (unsigned int i = 0; i < input.size(); i++)
		{
			KalmanTracker trk = KalmanTracker(input[i].box);
			trackers.push_back(trk);
		}
		return 0;
	}


	// 3.1. get predicted locations from existing trackers.
	predictedBoxes.clear();

	for (auto it = trackers.begin(); it != trackers.end();)
	{
		Rect_<float> pBox = (*it).predict();
		if (pBox.x >= 0 && pBox.y >= 0)
		{
			predictedBoxes.push_back(pBox);
			it++;
		}
		else
		{
			it = trackers.erase(it);
			//cerr << "Box invalid at frame: " << frame_count << endl;
		}
	}

	// 3.2. associate detections to tracked object (both represented as bounding boxes)
	// dets : detFrameData[fi]
	trkNum = predictedBoxes.size();
	detNum = input.size();

	iouMatrix.clear();
	iouMatrix.resize(trkNum, vector<double>(detNum, 0));

	for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
	{
		for (unsigned int j = 0; j < detNum; j++)
		{
			// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
			iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], input[j].box);
		}
	}

	// solve the assignment problem using hungarian algorithm.
	// the resulting assignment is [track(prediction) : detection], with len=preNum
	HungarianAlgorithm HungAlgo;
	assignment.clear();
	HungAlgo.Solve(iouMatrix, assignment);

	// find matches, unmatched_detections and unmatched_predictions
	unmatchedTrajectories.clear();
	unmatchedDetections.clear();
	allItems.clear();
	matchedItems.clear();

	if (detNum > trkNum) //	there are unmatched detections
	{
		for (unsigned int n = 0; n < detNum; n++)
			allItems.insert(n);

		for (unsigned int i = 0; i < trkNum; ++i)
			matchedItems.insert(assignment[i]);

		set_difference(allItems.begin(), allItems.end(),
			matchedItems.begin(), matchedItems.end(),
			insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
	}
	else if (detNum < trkNum) // there are unmatched trajectory/predictions
	{
		for (unsigned int i = 0; i < trkNum; ++i)
			if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
				unmatchedTrajectories.insert(i);
	}
	else
		;

	// filter out matched with low IOU
	matchedPairs.clear();
	for (unsigned int i = 0; i < trkNum; ++i)
	{
		if (assignment[i] == -1) // pass over invalid values
			continue;
		if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
		{
			unmatchedTrajectories.insert(i);
			unmatchedDetections.insert(assignment[i]);
		}
		else
			matchedPairs.push_back(cv::Point(i, assignment[i]));
	}

	///////////////////////////////////////
	// 3.3. updating trackers

	// update matched trackers with assigned detections.
	// each prediction is corresponding to a tracker
	int detIdx, trkIdx;
	for (unsigned int i = 0; i < matchedPairs.size(); i++)
	{
		trkIdx = matchedPairs[i].x;
		detIdx = matchedPairs[i].y;
		//trackers[trkIdx].m_type = input[detIdx].type;
		trackers[trkIdx].update(input[detIdx].box);
	}

	// create and initialise new trackers for unmatched detections
	for (auto umd : unmatchedDetections)
	{
		KalmanTracker tracker = KalmanTracker(input[umd].box);
		//tracker.m_type = vbox[umd].type;
		trackers.push_back(tracker);
	}

	// get trackers' output
	result.clear();
	for (auto it = trackers.begin(); it != trackers.end();)
	{
		if (((*it).m_time_since_update < 1) &&((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
		{
			if ((*it).m_isCounted == false)
		        {
			    (*it).m_id = total_id++;
			    (*it).m_isCounted = true;
		        }
			TrackingBox res;
			res.box = (*it).get_state();
			res.id = (*it).m_id + 1;
			result.push_back(res);
			it++;
		}
		else
			it++;

		// remove dead tracklet
		if (it != trackers.end() && (*it).m_time_since_update > max_age)
			it = trackers.erase(it);
	}

	return 1;
}
