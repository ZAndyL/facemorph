import sys
import dlib
import cv2
import numpy as np

def shape_to_landmarks(shape, dtype="int"):
    coords = [] 

    for i in range(0, 68):
        coords.append((shape.part(i).x, shape.part(i).y))

    return coords

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)

#calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    #create subdiv
    subdiv = cv2.Subdiv2D(rect)
    
    # Insert points into subdiv
    for p in points:
        subdiv.insert(p) 
    
    triangleList = subdiv.getTriangleList()
    
    delaunayTri = []
    
    for t in triangleList:   
        triangle_pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        ind = []
        for j in xrange(0, 3):
            for k in xrange(0, len(points)):                    
                if (abs(triangle_pts[j][0] - points[k][0]) < 1.0 and abs(triangle_pts[j][1] - points[k][1]) < 1.0):
                    ind.append(k)                            
        if len(ind) == 3:
            delaunayTri.append((ind[0], ind[1], ind[2]))
    
    return delaunayTri

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []


    for i in xrange(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(sys.argv[1])

# Process 1st image
image1 = cv2.imread(sys.argv[2])
rects = detector(image1, 1)
# Assume one face for now
face_rect = rects[0]
landmarks1 = shape_to_landmarks(predictor(image1, face_rect))
size = image1.shape
landmarks1.extend([(0,0), (0, size[0] - 1), (size[1] - 1, 0), (size[1] - 1, size[0] - 1)])
rect = (0, 0, size[1], size[0])

triangles = calculateDelaunayTriangles(rect, landmarks1)

# Process 2nd image
image2 = cv2.imread(sys.argv[3])
rects = detector(image2, 1)
face_rect = rects[0]
landmarks2 = shape_to_landmarks(predictor(image2, face_rect))
size = image2.shape
landmarks2.extend([(0,0), (0, size[0] - 1), (size[1] - 1, 0), (size[1] - 1, size[0] - 1)])

# Morph the images together
# Convert Mat to float data type
img1 = np.float32(image1)
img2 = np.float32(image2)

for i in range(1, 10):
    alpha = float(i)/10
    landmarks = []

    # Compute weighted average point coordinates
    for i in xrange(0, len(landmarks1)):
        x = ( 1 - alpha ) * landmarks1[i][0] + alpha * landmarks2[i][0]
        y = ( 1 - alpha ) * landmarks1[i][1] + alpha * landmarks2[i][1]
        landmarks.append((x,y))

    # Allocate space for final output
    imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

    for triangle in triangles:
        x = int(triangle[0])
        y = int(triangle[1])
        z = int(triangle[2])
        t1 = [landmarks1[x], landmarks1[y], landmarks1[z]]
        t2 = [landmarks2[x], landmarks2[y], landmarks2[z]]
        t = [ landmarks[x], landmarks[y], landmarks[z] ]

        # Morph one triangle at a time.
        morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

    cv2.imwrite(str(alpha)+".jpg", imgMorph)
