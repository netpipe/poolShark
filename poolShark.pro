QT       += widgets
TARGET    = sentry
TEMPLATE  = app
CONFIG   += c++14

INCLUDEPATH += /Users/macbook2015/Downloads/opencv-4.x/modules/video/include /Users/macbook2015/Downloads/opencv-4.x/modules/stitching/include /Users/macbook2015/Downloads/opencv-4.x/modules/photo/include /Users/macbook2015/Downloads/opencv-4.x/modules/objdetect/include /Users/macbook2015/Downloads/opencv-4.x/modules/ml/include /Users/macbook2015/Downloads/opencv-4.x/modules/imgproc/include /Users/macbook2015/Downloads/opencv-4.x/modules/videoio/include /Users/macbook2015/Downloads/opencv-4.x/modules/imgcodecs/include /Users/macbook2015/Downloads/opencv-4.x/modules/highgui/include /Users/macbook2015/Downloads/opencv-4.x/modules/dnn/include /Users/macbook2015/Downloads/opencv-4.x/modules/flann/include /Users/macbook2015/Downloads/opencv-4.x/modules/features2d/include /Users/macbook2015/Downloads/opencv-4.x/modules/calib3d/include /Users/macbook2015/Downloads/opencv-4.x/modules/core/include /usr/include/opencv4 /Users/macbook2015/Downloads/opencv-4.x/include  /Users/macbook2015/Downloads/opencv-4.x/build # adjust if needed (e.g. /usr/local/include)
LIBS        += -L/Users/macbook2015/Downloads/opencv-4.x/build/lib -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_highgui -lopencv_video


#LIBS += -L/usr/lib/x86_64-linux-gnu/ -lopencv_calib3d
#LIBS += -lopencv_core -lopencv_dnn -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml
#LIBS += -lopencv_objdetect -lopencv_photo -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_video
#LIBS += -lopencv_videoio -lopencv_videostab

SOURCES += main.cpp
