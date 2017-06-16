#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>

#include <math.h>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>


using namespace cv;
using namespace std;

// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat &warpImage, Mat &src, vector<Point2f> &srcTri, vector<Point2f> &dstTri)
{
    
    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform( srcTri, dstTri );
    
    // Apply the Affine Transform just found to the src image
    warpAffine( src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

// Warps and alpha blends triangular regions from img1 and img2 to img
void morphTriangle(Mat &img1, Mat &img2, Mat &img, vector<Point2f> &t1, vector<Point2f> &t2, vector<Point2f> &t, double alpha)
{
    
    // Find bounding rectangle for each triangle
    Rect r = boundingRect(t);
    Rect r1 = boundingRect(t1);
    Rect r2 = boundingRect(t2);
    
    // Offset points by left top corner of the respective rectangles
    vector<Point2f> t1Rect, t2Rect, tRect;
    vector<Point> tRectInt;
    for(int i = 0; i < 3; i++)
    {
        tRect.push_back( Point2f( t[i].x - r.x, t[i].y -  r.y) );
        tRectInt.push_back( Point(t[i].x - r.x, t[i].y - r.y) ); // for fillConvexPoly
        
        t1Rect.push_back( Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
        t2Rect.push_back( Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
    }
    
    // Get mask by filling triangle
    Mat mask = Mat::zeros(r.height, r.width, CV_32FC3);
    fillConvexPoly(mask, tRectInt, Scalar(1.0, 1.0, 1.0), 16, 0);
    
    // Apply warpImage to small rectangular patches
    Mat img1Rect, img2Rect;
    img1(r1).copyTo(img1Rect);
    img2(r2).copyTo(img2Rect);
    
    Mat warpImage1 = Mat::zeros(r.height, r.width, img1Rect.type());
    Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type());
    
    applyAffineTransform(warpImage1, img1Rect, t1Rect, tRect);
    applyAffineTransform(warpImage2, img2Rect, t2Rect, tRect);
    
    // Alpha blend rectangular patches
    Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2;
    
    // Copy triangular region of the rectangular patch to the output image
    multiply(imgRect,mask, imgRect);
    multiply(img(r), Scalar(1.0,1.0,1.0) - mask, img(r));
    img(r) = img(r) + imgRect;
    
    
}

vector<Point> shapeToLandmarks(dlib::full_object_detection shape) {
    vector<Point> list;
    for(int i = 0; i < 68; i++) {
        list.push_back(Point(shape.part(i).x(), shape.part(i).y()));
    }
    return list;
}

vector<tuple<int, int, int>> calculateDelaunayTriangles(Rect rect, vector<Point> points) {
    Subdiv2D subdiv(rect);
    for (Point p : points) {
        subdiv.insert(p);
    }
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<tuple<int, int, int>> dTriangles;
    
    for( int i = 0; i < triangleList.size(); i++ )
    {
        Vec6f t = triangleList[i];
        vector<Point> triangle = {Point(cvRound(t[0]), cvRound(t[1])),
                                  Point(cvRound(t[2]), cvRound(t[3])),
                                  Point(cvRound(t[4]), cvRound(t[5]))};
        vector<int> indices;
        for (int i = 0; i < 3; i++) {
            Point p = triangle[i];
            for (int j = 0; j < points.size(); j++) {
                if (p.x == points[j].x && p.y == points[j].y) {
                    indices.push_back(j);
                }
            }
        }
        if (indices.size() == 3) {
            dTriangles.push_back(make_tuple(indices[0], indices[1], indices[2]));
        }

    }
    return dTriangles;
}

void deserializePacked(dlib::shape_predictor &sp) {
    
    int idx;
    int nc = 0, nr = 0;
    
    // load the shape predictor from the packed file:
    auto file = fopen("sp_packed.dat", "rb");
    
    cout << "reading initial shape" << endl;
    
    // read in the initial shape:
    sp.initial_shape.set_size(136, 1);
    float floatBuf[136];
    fread(floatBuf, 4, 136, file);
    for(int r = 0; r < 136; r++)
    {
        sp.initial_shape(r, 0) = floatBuf[r];
    }
    
    cout << "reading forests" << endl;
    
    // read in the forests:
    sp.forests.resize(15);
    for(idx = 0; idx < 15; idx++)
    {
        sp.forests[idx].resize(500);
        int tree;
        
        for(tree = 0; tree < 500; tree++)
        {
            // cout << "reading splits" << endl;
            
            // read in the splits:
            sp.forests[idx][tree].splits.resize(15);
            for(int split = 0; split < 15; split++)
            {
                unsigned int idx1, idx2;
                float thresh;
                fread(&idx1, 4, 1, file);
                fread(&idx2, 4, 1, file);
                fread(&thresh, 4, 1, file);
                
                sp.forests[idx][tree].splits[split].idx1 = (unsigned long)idx1;
                sp.forests[idx][tree].splits[split].idx2 = (unsigned long)idx2;
                sp.forests[idx][tree].splits[split].thresh = thresh;
            }
            
            // cout << "reading leaf_values" << endl;
            
            sp.forests[idx][tree].leaf_values.resize(16);
            int leaf;
            for(leaf = 0; leaf < 16; leaf++)
            {
                sp.forests[idx][tree].leaf_values[leaf].set_size(136, 1);
                
                float leafmin, step;
                fread(&leafmin, 4, 1, file);
                fread(&step, 4, 1, file);
                
                unsigned char quantized[136];
                fread(quantized, 136, 1, file);
                int qidx = 0;
                for(int r = 0; r < 136; r++)
                {
                    int q = (int)quantized[r] & 255;
                    float value = leafmin + ((float)q * step);
                    sp.forests[idx][tree].leaf_values[leaf](r, 0) = value;
                }
            }
        }
    }
    
    cout << "reading anchor_idx" << endl;
    
    // read in anchor_idx:
    sp.anchor_idx.resize(15);
    for(idx = 0; idx < 15; idx++)
    {
        sp.anchor_idx[idx].resize(500);
        for(int aidx = 0; aidx < 500; aidx++)
        {
            unsigned int val;
            fread(&val, 4, 1, file);
            sp.anchor_idx[idx][aidx] = (unsigned long)val;			
        }
    }
    
    cout << "reading deltas" << endl;
    
    // read in the deltas:
    sp.deltas.resize(15);
    for(idx = 0; idx < 15; idx++)
    {
        sp.deltas[idx].resize(500);
        for(int didx = 0; didx < 500; didx++)
        {
            float x, y;
            fread(&x, 4, 1, file);
            fread(&y, 4, 1, file);
            sp.deltas[idx][didx].x() = x;
            sp.deltas[idx][didx].y() = y;
        }
    }
    
    cout << "closing file" << endl;
    
    fclose(file);
}

int main( int argc, char** argv)
{
    
    string filename1("hillary_clinton.jpg");
    string filename2("ted_cruz.jpg");
    
    //alpha controls the degree of morph
    double alpha = 0.5;
    
    //Read input images
    Mat img1 = imread(filename1);
    Mat img2 = imread(filename2);
    
    
    //empty average image
    Mat imgMorph = Mat::zeros(img1.size(), CV_32FC3);
    
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    deserializePacked(predictor);
    
    Size size = img1.size();
    Point boundaries[] = {Point(0, 0), Point(0, size.height - 1), Point(size.width - 1, 0), Point(size.width - 1, size.height - 1)};
    
    dlib::cv_image<dlib::bgr_pixel> cimg1(img1);
    auto rects = detector(cimg1);
    auto landmarks1 = shapeToLandmarks(predictor(cimg1, rects[0]));
    for (auto b : boundaries) {
        landmarks1.push_back(b);
    }
    auto dTriangles = calculateDelaunayTriangles(Rect(0, 0, size.width, size.height), landmarks1);
    
    dlib::cv_image<dlib::bgr_pixel> cimg2(img2);
    rects = detector(cimg2);
    auto landmarks2 = shapeToLandmarks(predictor(cimg2, rects[0]));
    for (auto b : boundaries) {
        landmarks2.push_back(b);
    }
    
    vector<Point> landmarks;
    
    img1.convertTo(img1, CV_32F);
    img2.convertTo(img2, CV_32F);
    
    //compute weighted average point coordinates
    for(int i = 0; i < landmarks1.size(); i++)
    {
        float x, y;
        x = (1 - alpha) * (float)landmarks1[i].x + alpha * (float)landmarks2[i].x;
        y =  ( 1 - alpha ) * (float)landmarks1[i].y + alpha * (float)landmarks2[i].y;

        landmarks.push_back(Point2f(x,y));
    }
    
    for (tuple<int, int, int> triangle : dTriangles)
    {
        int x = get<0>(triangle);
        int y = get<1>(triangle);
        int z = get<2>(triangle);
        
        // Triangles
        vector<Point2f> t1, t2, t;

        // Triangle corners for image 1.
        t1.push_back( landmarks1[x] );
        t1.push_back( landmarks1[y] );
        t1.push_back( landmarks1[z] );

        // Triangle corners for image 2.
        t2.push_back( landmarks2[x] );
        t2.push_back( landmarks2[y] );
        t2.push_back( landmarks2[z] );

        // Triangle corners for morphed image.
        t.push_back( landmarks[x] );
        t.push_back( landmarks[y] );
        t.push_back( landmarks[z] );
        
        morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha);
    }
    
    imwrite("morphed.jpg", imgMorph);
    
    return 0;
}
