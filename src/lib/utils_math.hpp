#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <iostream>

using namespace dlib;
using namespace std;

float stddev(float data[], int len);
float mean(float data[], int len);

std::vector<full_object_detection> process(array2d<rgb_pixel> &img, shape_predictor &sp, frontal_face_detector &detector)
{
  std::vector<dlib::rectangle> dets = detector(img); // Number of faces detected
  std::vector<full_object_detection> shapes;
  // float ear;
  for (unsigned long j = 0; j < dets.size(); ++j)
  {
    full_object_detection shape = sp(img, dets[j]); // face landmark
    // ear = eye_aspect_ratio(shape);

    shapes.push_back(shape);
  }
  return shapes;
}

float distanceCalculate(point &p1, point &p2)
{
  float x = p1.x() - p2.x(); // calculating number to square in next step
  float y = p1.y() - p2.y();
  return sqrt(pow(x, 2) + pow(y, 2)); // calculating Euclidean distance
}

float eye_aspect_ratio(full_object_detection &shape)
{
  float r_ear = (distanceCalculate(shape.part(43), shape.part(47)) + distanceCalculate(shape.part(44), shape.part(46))) / (2 * distanceCalculate(shape.part(42), shape.part(45)));
  float l_ear = (distanceCalculate(shape.part(37), shape.part(41)) + distanceCalculate(shape.part(38), shape.part(40))) / (2 * distanceCalculate(shape.part(36), shape.part(39)));
  return (l_ear + r_ear) / 2;
}

float mouth_aspect_ratio(full_object_detection &shape)
{
  float a = distanceCalculate(shape.part(61), shape.part(67));
  float b = distanceCalculate(shape.part(62), shape.part(66));
  float c = distanceCalculate(shape.part(63), shape.part(65));
  float d = distanceCalculate(shape.part(60), shape.part(64));
  
  return (a + b + c) / (3 * d);
}

std::vector<float> threshold_calculate(std::vector<full_object_detection> &shapes){
  int size = shapes.size();
  float a[size], b[size], c[size], d[size];
  std::vector<float> threshold;
  std::vector<float> stddevs;

  for(int i=0; i<size; i++){
    a[i] = eye_aspect_ratio(shapes.at(i));
    b[i] = mouth_aspect_ratio(shapes.at(i));
    c[i] = shapes.at(i).part(30).x();
    d[i] = shapes.at(i).part(30).y();    
  }
  
  threshold.push_back(mean(a, size)-0.03);
  threshold.push_back(mean(b, size)+0.5);
  threshold.push_back(mean(c, size));
  threshold.push_back(mean(d, size));

  // threshold.push_back(stddev(a, size));
  // threshold.push_back(stddev(b, size));
  // threshold.push_back(stddev(c, size));
  // threshold.push_back(stddev(d, size));
  // cout << mean(b, size) << endl;
  // cout << "EYE_AR_THRESH " << threshold.at(0) <<endl;
  // cout << "MOUTH_AR_THRESH " << threshold.at(1) <<endl;
  // cout << "HEAD_X_THRESH " << threshold.at(2) <<endl;
  // cout << "HEAD_Y_THRESH " << threshold.at(3) <<endl;
  return threshold;
}
/*
void thresholding(float y[], int signals[], int lag, float threshold, float influence) {
    memset(signals, 0, sizeof(int) * SAMPLE_LENGTH);
    float filteredY[SAMPLE_LENGTH];
    memcpy(filteredY, y, sizeof(float) * SAMPLE_LENGTH);
    float avgFilter[SAMPLE_LENGTH];
    float stdFilter[SAMPLE_LENGTH];

    avgFilter[lag - 1] = mean(y, lag);
    stdFilter[lag - 1] = stddev(y, lag);

    for (int i = lag; i < SAMPLE_LENGTH; i++) {
        if (fabsf(y[i] - avgFilter[i-1]) > threshold * stdFilter[i-1]) {
            if (y[i] > avgFilter[i-1]) {
                signals[i] = 1;
            } else {
                signals[i] = -1;
            }
            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1];
        } else {
            signals[i] = 0;
        }
        avgFilter[i] = mean(filteredY + i-lag, lag);
        stdFilter[i] = stddev(filteredY + i-lag, lag);
    }
}
*/

float mean(float data[], int len) {
    float sum = 0.0, mean = 0.0;


    for(int i=0; i<len; ++i) {
        sum += data[i];
    }

    mean = sum/len;
    return mean;
}

float stddev(float data[], int len) {
    float the_mean = mean(data, len);
    float standardDeviation = 0.0;

    int i;
    for(i=0; i<len; ++i) {
        standardDeviation += pow(data[i] - the_mean, 2);
    }

    return sqrt(standardDeviation/len);
}
