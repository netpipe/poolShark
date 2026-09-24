/* ============================================================================
 *  PoolShark — single-file demo  (Qt 5.12 + OpenCV 4, C++11)
 *  AR pool-shot assistant for camera-equipped "recording glasses".
 *
 *  Features
 *  --------
 *   - Live capture from any UVC camera (your glasses), optional mirroring
 *   - Table felt detection (green OR blue) to constrain the search area
 *   - Multi-ball detection + persistent ID tracking while you move your head
 *     (sparse optical-flow global motion compensation between frames)
 *   - Cue-stick detection:
 *       * by adjustable HSV colour (hue centre/window, min sat, min value)
 *       * by shape only (long straight edge clusters -> works for ANY colour)
 *       * one-click colour sampling straight from the video
 *   - Aim solver: cue axis -> cue ball -> ghost ball -> first object ball,
 *     object-ball trajectory + cue-ball deflection line, cut angle readout
 *   - Overlay rendering + optional MJPG recording of the AR video
 *
 *  Build — create poolshark.pro next to this file:
 *
 *      QT       += core gui widgets
 *      CONFIG   += c++11
 *      SOURCES  += poolshark.cpp
 *      # point these at your OpenCV 4 install, e.g.:
 *      INCLUDEPATH += /usr/local/include/opencv4
 *      LIBS += -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_videoio
 *      # (no highgui on purpose: avoids double-Qt linkage problems)
 *
 *      $ qmake poolshark.pro && make
 *
 *  Notes / limitations (it is a demo):
 *   - No camera calibration: wide-angle glass lenses distort geometry.
 *   - Tracking is 2D image-space; very fast rotation may drop IDs briefly.
 *   - Physics = flat table, frictionless ghost-ball model.
 *   - No Q_OBJECT macros on purpose -> the file needs no moc step.
 * ==========================================================================*/

#include <QApplication>
#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QComboBox>
#include <QSpinBox>
#include <QSlider>
#include <QGroupBox>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTimer>
#include <QThread>
#include <QMutex>
#include <QMutexLocker>
#include <QElapsedTimer>
#include <QMouseEvent>
#include <QResizeEvent>
#include <QCloseEvent>
#include <QDateTime>
#include <QImage>
#include <QPixmap>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

/* ========================================================================== *
 *  Small helpers
 * ========================================================================== */

static QImage matToQImage(const cv::Mat &m)
{
    cv::Mat rgb;
    cv::cvtColor(m, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, (int)rgb.step,
                  QImage::Format_RGB888).copy();
}

static float medianFloat(std::vector<float> &v)
{
    if (v.empty()) return 0.f;
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

/* inRange on a hue interval that may wrap around 180 (red zone). */
static void inRangeHue(const cv::Mat &hsv, int hLo, int hHi,
                       int sMin, int vMin, cv::Mat &out)
{
    hLo = ((hLo % 180) + 180) % 180;
    hHi = ((hHi % 180) + 180) % 180;
    if (hLo <= hHi) {
        cv::inRange(hsv, cv::Scalar(hLo, sMin, vMin),
                    cv::Scalar(hHi, 255, 255), out);
    } else {
        cv::Mat a, b;
        cv::inRange(hsv, cv::Scalar(hLo, sMin, vMin),
                    cv::Scalar(179, 255, 255), a);
        cv::inRange(hsv, cv::Scalar(0, sMin, vMin),
                    cv::Scalar(hHi, 255, 255), b);
        out = a | b;
    }
}

/* Rough pool-ball colour naming from average HSV. */
static void classifyHsv(const cv::Vec3b &c, std::string &label,
                        bool &isCue, cv::Scalar &bgr)
{
    using cv::Scalar;
    int h = c[0], s = c[1], v = c[2];
    isCue = false;
    if (v >= 165 && s <= 70) { label = "CUE";   isCue = true; bgr = Scalar(255,255,255); return; }
    if (v <= 80)             { label = "BLACK";                 bgr = Scalar( 30, 30, 30); return; }
    if (s < 60)              { label = "GREY";                  bgr = Scalar(160,160,160); return; }
    if (h < 8 || h >= 172)   { label = "RED";                   bgr = Scalar( 40, 40,230); }
    else if (h < 20)         { label = "ORANGE";                bgr = Scalar( 30,140,255); }
    else if (h < 34)         { label = "YELLOW";                bgr = Scalar( 40,220,255); }
    else if (h < 80)         { label = "GREEN";                 bgr = Scalar( 60,200, 60); }
    else if (h < 130)        { label = "BLUE";                  bgr = Scalar(230,120, 40); }
    else                     { label = "PURPLE";                bgr = Scalar(200, 60,160); }
}

static cv::Point2f extendToRect(const cv::Point2f &o, const cv::Point2f &d,
                                cv::Size sz)
{
    float t = 1e6f;
    if      (d.x >  1e-6f) t = std::min(t, ((float)sz.width  - 1.f - o.x) /  d.x);
    else if (d.x < -1e-6f) t = std::min(t, (-o.x) / -d.x * -1.f);
    if      (d.y >  1e-6f) t = std::min(t, ((float)sz.height - 1.f - o.y) /  d.y);
    else if (d.y < -1e-6f) t = std::min(t, (-o.y) / -d.y * -1.f);
    if (t < 0.f)    t = 0.f;
    if (t > 1e5f)   t = 1e5f;
    return o + d * t;
}

static void dashedLine(cv::Mat &img, cv::Point2f a, cv::Point2f b,
                       const cv::Scalar &col, int thick, float dashLen = 10.f)
{
    float L = (float)cv::norm(b - a);
    if (L < 1.f) return;
    cv::Point2f u = (b - a) * (1.f / L);
    float x = 0.f;
    while (x < L) {
        float x1 = std::min(x + dashLen, L);
        cv::line(img, a + u * x, a + u * x1, col, thick, cv::LINE_AA);
        x = x1 + dashLen;
    }
}

static void dashedCircle(cv::Mat &img, cv::Point2f c, float r,
                         const cv::Scalar &col, int thick, int segs = 24)
{
    for (int i = 0; i < segs; i += 2) {
        float a0 = (float)i       / segs * 2.f * (float)CV_PI;
        float a1 = (float)(i + 1) / segs * 2.f * (float)CV_PI;
        cv::Point2f p0(c.x + r * std::cos(a0), c.y + r * std::sin(a0));
        cv::Point2f p1(c.x + r * std::cos(a1), c.y + r * std::sin(a1));
        cv::line(img, p0, p1, col, thick, cv::LINE_AA);
    }
}

/* ========================================================================== *
 *  Data types
 * ========================================================================== */

struct Params
{
    bool  flip         = false;
    int   maxWidth     = 1024;   // processing width (speed)
    int   cueMode      = 0;      // 0 = colour, 1 = shape (any colour)
    bool  shapeFallback = true;  // try shape mode when colour mode fails
    int   hueC = 22, hueW = 16, sMin = 40, vMin = 70; // cue colour gate
    float minBallR     = 9.f;    // px in processing resolution
    bool  showBalls = true, showCue = true, showAim = true, showTable = true;
};

struct BallDet
{
    cv::Point2f c;  float r = 0;  float fill = 0;
    cv::Vec3b   hsv;
    std::string label;
    cv::Scalar  bgr;
    bool        isCue = false;
};

struct BallTrack
{
    int         id = 0;
    cv::Point2f pos, vel;            // px, px/frame
    float       radius = 10.f;
    float       hE = 0, sE = 0, vE = 0;   // EMA of HSV
    std::string label = "?";
    cv::Scalar  bgr = cv::Scalar(200, 200, 200);
    bool        isCue = false;
    int         hits = 0, misses = 0;
    bool confirmed() const { return hits >= 2; }
};

struct CueInfo
{
    bool        found = false, byColor = true;
    cv::Point2f tip, butt;
    double      len = 0;
};

struct AimSolution
{
    bool        valid = false;
    bool        haveCueBall = false;
    int         targetIdx = -1;
    cv::Point2f dir, tip, rayEnd, ghost, objDir, cueDir, cueBallPos;
    float       ghostR = 10.f, cutDeg = 0.f, deflMag = 0.f;
};

struct Analysis
{
    cv::Mat                    frame;
    cv::Mat                    tableMask, tableMaskD, ballBlock;
    std::vector<cv::Point>     tableContour;
    cv::Point2f                tableCenter;
    cv::Point2f                shift;        // estimated head motion / frame
    std::vector<BallDet>       ballDets;
    std::vector<BallTrack>     tracks;
    int                        cueBallIdx = -1;
    CueInfo                    cue;
    AimSolution                aim;
};

/* ========================================================================== *
 *  PoolEngine — all the computer vision, keeps state between frames
 * ========================================================================== */

class PoolEngine
{
public:
    void reset() { tracks.clear(); prevGray.release(); nextId = 1; }
    void process(const cv::Mat &frame, const Params &p, Analysis &A);

private:
    void detectTable(const cv::Mat &hsv, Analysis &A);
    cv::Point2f estimateMotion(const cv::Mat &gray, const cv::Mat &mask);
    void detectBalls(const cv::Mat &hsv, Analysis &A, const Params &p);
    void updateTracks(Analysis &A, cv::Point2f shift);
    void detectCue(const cv::Mat &blur, const cv::Mat &hsv,
                   Analysis &A, const Params &p);
    bool cueByColor(const cv::Mat &hsv, const Analysis &A,
                    const Params &p, CueInfo &C);
    bool cueByShape(const cv::Mat &blur, const Analysis &A, CueInfo &C);
    void solveAim(Analysis &A, const Params &p);
    int  quickCueTrack(const Analysis &A) const;

    std::vector<BallTrack> tracks;
    cv::Mat                prevGray;
    int                    nextId = 1;
};

/* ---- main per-frame pipeline ------------------------------------------- */
void PoolEngine::process(const cv::Mat &frameIn, const Params &p, Analysis &A)
{
    using namespace cv;
    if (frameIn.empty()) return;

    Mat frame = frameIn;
    if (p.flip) flip(frame, frame, 1);
    if (frame.cols > p.maxWidth) {
        double s = (double)p.maxWidth / frame.cols;
        resize(frame, frame, Size(), s, s, INTER_AREA);
    }
    A.frame = frame;

    Mat gray, blur, hsv;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(5, 5), 1.1);
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    detectTable(hsv, A);                       // felt region
    A.shift = estimateMotion(gray, A.tableMask); // head movement compensation
    detectBalls(hsv, A, p);                    // per-frame ball candidates
    updateTracks(A, A.shift);                  // persistent IDs
    A.tracks = tracks;

    /* mask of confirmed balls — keeps same-colour balls out of cue mask */
    A.ballBlock = Mat::zeros(frame.size(), CV_8U);
    for (const auto &t : tracks)
        if (t.confirmed())
            circle(A.ballBlock, t.pos, (int)(t.radius + 5), Scalar(255), -1);

    detectCue(blur, hsv, A, p);                // stick: colour or shape
    solveAim(A, p);                            // ghost ball + trajectories

    prevGray = gray.clone();
}

/* ---- table felt -------------------------------------------------------- */
void PoolEngine::detectTable(const cv::Mat &hsv, Analysis &A)
{
    using namespace cv;
    Mat g, b, m;
    inRange(hsv, Scalar(35, 45, 40), Scalar( 85, 255, 255), g); // green felt
    inRange(hsv, Scalar(95, 45, 40), Scalar(128, 255, 255), b); // blue felt
    m = g | b;
    morphologyEx(m, m, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(15,15)));
    morphologyEx(m, m, MORPH_OPEN,  getStructuringElement(MORPH_ELLIPSE, Size(7,7)));

    std::vector<std::vector<Point>> cs;
    findContours(m, cs, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    double best = 0; int bi = -1;
    for (size_t i = 0; i < cs.size(); ++i) {
        double a = contourArea(cs[i]);
        if (a > best) { best = a; bi = (int)i; }
    }

    double need = 0.12 * m.total();
    if (bi >= 0 && best > need) {
        A.tableMask = Mat::zeros(m.size(), CV_8U);
        drawContours(A.tableMask, cs, bi, Scalar(255), -1);
        approxPolyDP(cs[bi], A.tableContour, 10.0, true);
        Moments mu = moments(cs[bi]);
        A.tableCenter = (mu.m00 > 0)
            ? Point2f((float)(mu.m10 / mu.m00), (float)(mu.m01 / mu.m00))
            : Point2f(m.cols / 2.f, m.rows / 2.f);
    } else {
        A.tableMask = Mat(m.size(), CV_8U, Scalar(255)); // fallback: whole frame
        A.tableContour.clear();
        A.tableCenter = Point2f(m.cols / 2.f, m.rows / 2.f);
    }
    dilate(A.tableMask, A.tableMaskD, getStructuringElement(MORPH_ELLIPSE, Size(25,25)));
}

/* ---- global motion (head movement) via sparse LK flow ------------------- */
cv::Point2f PoolEngine::estimateMotion(const cv::Mat &gray, const cv::Mat &mask)
{
    using namespace cv;
    if (prevGray.empty() || prevGray.size() != gray.size()) return Point2f(0, 0);

    Mat fm;
    erode(mask, fm, getStructuringElement(MORPH_ELLIPSE, Size(21, 21)));
    std::vector<Point2f> p0, p1;
    goodFeaturesToTrack(prevGray, p0, 160, 0.02, 14.0, fm);
    if (p0.size() < 10) return Point2f(0, 0);

    std::vector<uchar> st; std::vector<float> er;
    calcOpticalFlowPyrLK(prevGray, gray, p0, p1, st, er, Size(21, 21), 3);

    std::vector<float> dx, dy;
    for (size_t i = 0; i < p0.size(); ++i)
        if (st[i] && er[i] < 8.f) {
            dx.push_back(p1[i].x - p0[i].x);
            dy.push_back(p1[i].y - p0[i].y);
        }
    if (dx.size() < 8) return Point2f(0, 0);

    float mx = medianFloat(dx), my = medianFloat(dy);
    if (std::fabs(mx) > 60.f) mx = 0;   // implausible jump -> ignore
    if (std::fabs(my) > 60.f) my = 0;
    return Point2f(mx, my);
}

/* ---- per-frame ball candidates ------------------------------------------ */
void PoolEngine::detectBalls(const cv::Mat &hsv, Analysis &A, const Params &p)
{
    using namespace cv;
    Mat white, colored, dark, m;
    inRange(hsv, Scalar(0,   0, 170), Scalar(180,  65, 255), white);  // cue ball
    inRange(hsv, Scalar(0,  95,  80), Scalar(180, 255, 255), colored);// object balls
    inRange(hsv, Scalar(0,   0,  15), Scalar(180, 170,  80), dark);   // 8-ball
    m = white | colored | dark;
    if (!A.tableMaskD.empty()) m = m & A.tableMaskD;

    morphologyEx(m, m, MORPH_OPEN,  getStructuringElement(MORPH_RECT,    Size(3,3)));
    morphologyEx(m, m, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));

    std::vector<std::vector<Point>> cs;
    findContours(m, cs, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    float rmin = p.minBallR, rmax = p.minBallR * 3.4f;
    std::vector<BallDet> dets;

    for (const auto &c : cs) {
        double area = contourArea(c);
        if (area < 20) continue;
        Point2f cc; float r;
        minEnclosingCircle(c, cc, r);
        if (r < rmin || r > rmax) continue;
        double fill = area / (CV_PI * r * r);
        if (fill < 0.52 || fill > 1.10) continue;   // roundness gate

        /* sample average HSV inside the blob for classification */
        Rect bb = boundingRect(c) & Rect(0, 0, hsv.cols, hsv.rows);
        if (bb.width < 4 || bb.height < 4) continue;
        Mat rm = Mat::zeros(bb.size(), CV_8U);
        circle(rm, cc - Point2f((float)bb.x, (float)bb.y),
               std::max(2, (int)(r * 0.55f)), Scalar(255), -1);
        Scalar mh = mean(hsv(bb), rm);

        BallDet d;
        d.c = cc; d.r = r; d.fill = (float)fill;
        d.hsv = Vec3b(saturate_cast<uchar>(mh[0]),
                      saturate_cast<uchar>(mh[1]),
                      saturate_cast<uchar>(mh[2]));
        classifyHsv(d.hsv, d.label, d.isCue, d.bgr);
        dets.push_back(d);
    }

    /* NMS — keep the roundest when blobs overlap */
    std::sort(dets.begin(), dets.end(),
              [](const BallDet &a, const BallDet &b) { return a.fill > b.fill; });
    for (const auto &d : dets) {
        bool dup = false;
        for (const auto &k : A.ballDets)
            if (norm(d.c - k.c) < 07.0f / 10.f * (d.r + k.r)) { dup = true; break; }
        if (!dup) A.ballDets.push_back(d);
    }
}

/* ---- greedy nearest-neighbour tracker with motion compensation ---------- */
void PoolEngine::updateTracks(Analysis &A, cv::Point2f shift)
{
    using namespace cv;
    auto &T = tracks;
    const auto &D = A.ballDets;

    std::vector<char> usedT(T.size(), 0), usedD(D.size(), 0);
    struct M { float d; size_t t, di; };
    std::vector<M> ms;

    for (size_t t = 0; t < T.size(); ++t) {
        Point2f pred = T[t].pos + T[t].vel + shift;
        float gate = std::max(22.f, 2.4f * T[t].radius);
        for (size_t d = 0; d < D.size(); ++d) {
            float dist = (float)norm(pred - D[d].c);
            if (dist < gate) ms.push_back(M{dist, t, d});
        }
    }
    std::sort(ms.begin(), ms.end(), [](const M &a, const M &b) { return a.d < b.d; });

    for (const auto &m : ms) {
        if (usedT[m.t] || usedD[m.di]) continue;
        usedT[m.t] = usedD[m.di] = 1;
        BallTrack &tk = T[m.t];
        const BallDet &dt = D[m.di];

        Point2f pred = tk.pos + tk.vel + shift;
        tk.vel    = 0.65f * tk.vel + 0.35f * (dt.c - pred);
        tk.pos    = 0.35f * pred   + 0.65f * dt.c;
        tk.radius = 0.7f  * tk.radius + 0.3f * dt.r;

        float dh = dt.hsv[0] - tk.hE;                  // hue wrap guard
        if (std::fabs(dh) < 90.f) tk.hE = 0.7f * tk.hE + 0.3f * dt.hsv[0];
        tk.sE = 0.7f * tk.sE + 0.3f * dt.hsv[1];
        tk.vE = 0.7f * tk.vE + 0.3f * dt.hsv[2];

        auto cl = [](float v) { return (uchar)std::max(0.f, std::min(255.f, v)); };
        Vec3b avg(cl(tk.hE), cl(tk.sE), cl(tk.vE));
        classifyHsv(avg, tk.label, tk.isCue, tk.bgr);
        tk.hits++; tk.misses = 0;
    }

    /* missed tracks coast along with head motion */
    for (size_t t = 0; t < T.size(); ++t) if (!usedT[t]) {
        T[t].pos += T[t].vel + shift;
        T[t].vel *= 0.8f;
        T[t].misses++;
    }
    T.erase(std::remove_if(T.begin(), T.end(),
                           [](const BallTrack &b) { return b.misses > 12; }),
            T.end());

    /* spawn new tracks */
    for (size_t d = 0; d < D.size(); ++d) if (!usedD[d]) {
        BallTrack nt;
        nt.id = nextId++;
        nt.pos = D[d].c; nt.radius = D[d].r;
        nt.hE = D[d].hsv[0]; nt.sE = D[d].hsv[1]; nt.vE = D[d].hsv[2];
        nt.label = D[d].label; nt.bgr = D[d].bgr; nt.isCue = D[d].isCue;
        nt.hits = 1;
        T.push_back(nt);
    }
}

/* ---- cue stick ---------------------------------------------------------- */
int PoolEngine::quickCueTrack(const Analysis &A) const
{
    int best = -1, bestHits = 1;
    for (size_t i = 0; i < A.tracks.size(); ++i)
        if (A.tracks[i].isCue && A.tracks[i].confirmed() && A.tracks[i].hits > bestHits)
            bestHits = A.tracks[i].hits, best = (int)i;
    return best;
}

void PoolEngine::detectCue(const cv::Mat &blur, const cv::Mat &hsv,
                           Analysis &A, const Params &p)
{
    CueInfo C;
    bool ok = false;
    if (p.cueMode == 0) ok = cueByColor(hsv, A, p, C);
    if (!ok && (p.cueMode == 1 || p.shapeFallback)) ok = cueByShape(blur, A, C);

    if (ok) {   /* tip = endpoint nearest to the cue ball (or table centre) */
        cv::Point2f anchor = A.tableCenter;
        int ci = quickCueTrack(A);
        if (ci >= 0) anchor = A.tracks[ci].pos;
        if (cv::norm(C.tip - anchor) > cv::norm(C.butt - anchor))
            std::swap(C.tip, C.butt);
    }
    A.cue = C;
}

bool PoolEngine::cueByColor(const cv::Mat &hsv, const Analysis &A,
                            const Params &p, CueInfo &C)
{
    using namespace cv;
    Mat mask;
    inRangeHue(hsv, p.hueC - p.hueW, p.hueC + p.hueW, p.sMin, p.vMin, mask);
    if (!A.ballBlock.empty()) {
        Mat nb; bitwise_not(A.ballBlock, nb);
        bitwise_and(mask, nb, mask);
    }
    morphologyEx(mask, mask, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(9,9)));
    morphologyEx(mask, mask, MORPH_OPEN,  getStructuringElement(MORPH_ELLIPSE, Size(3,3)));

    std::vector<std::vector<Point>> cs;
    findContours(mask, cs, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double diag = std::hypot((double)mask.cols, (double)mask.rows);
    double best = 0; int bi = -1;
    for (size_t i = 0; i < cs.size(); ++i) {
        RotatedRect rr = minAreaRect(cs[i]);
        float L = std::max(rr.size.width, rr.size.height);
        float W = std::max(1.f, std::min(rr.size.width, rr.size.height));
        if (L < diag * 0.15) continue;
        double aspect = L / W;
        if (aspect < 5.0) continue;                 // must be long & skinny
        double score = L * std::min(aspect, 60.0);
        if (score > best) { best = score; bi = (int)i; }
    }
    if (bi < 0) return false;

    Vec4f ln;
    fitLine(cs[bi], ln, DIST_L2, 0, 0.01, 0.01);
    Point2f dir(ln[0], ln[1]), base(ln[2], ln[3]);
    float smin = 1e9f, smax = -1e9f;
    for (const auto &pt : cs[bi]) {
        float s = (Point2f(pt) - base).dot(dir);
        smin = std::min(smin, s);
        smax = std::max(smax, s);
    }
    C.found = true; C.byColor = true;
    C.tip  = base + dir * smin;
    C.butt = base + dir * smax;
    C.len  = smax - smin;
    return true;
}

bool PoolEngine::cueByShape(const cv::Mat &blur, const Analysis &A, CueInfo &C)
{
    using namespace cv;
    Mat e;
    Canny(blur, e, 50, 150, 3);
    if (!A.ballBlock.empty()) {
        Mat nb; bitwise_not(A.ballBlock, nb);
        bitwise_and(e, nb, e);
    }
    double diag = std::hypot((double)e.cols, (double)e.rows);

    std::vector<Vec4i> segs;
    HoughLinesP(e, segs, 1, CV_PI / 180.0,
                std::max(40, (int)(diag / 22)),   // vote threshold
                diag * 0.18,                      // min line length
                18);                              // max gap to bridge
    if (segs.empty()) return false;

    struct Seg { Point2f a, b, dir; float len, ang; };
    std::vector<Seg> ls;
    for (const auto &v : segs) {
        Seg s;
        s.a = Point2f((float)v[0], (float)v[1]);
        s.b = Point2f((float)v[2], (float)v[3]);
        Point2f d = s.b - s.a;
        s.len = (float)norm(d);
        if (s.len < 2.f) continue;
        s.dir = d * (1.f / s.len);
        s.ang = std::atan2(d.y, d.x);
        ls.push_back(s);
    }

    /* cluster near-collinear segments, keep the longest stick-like group */
    float bestScore = 0; Point2f bA, bB; bool found = false;
    for (size_t i = 0; i < ls.size(); ++i) {
        float score = 0;
        std::vector<Point2f> pts;
        for (size_t j = 0; j < ls.size(); ++j) {
            float da = std::fabs(std::remainder(ls[j].ang - ls[i].ang, (float)CV_PI));
            if (da > 5.f * (float)CV_PI / 180.f) continue;
            Point2f mid = (ls[j].a + ls[j].b) * 0.5f;
            float dist = std::fabs((mid - ls[i].a).cross(ls[i].dir));
            if (dist > 10.f) continue;
            score += ls[j].len;
            pts.push_back(ls[j].a);
            pts.push_back(ls[j].b);
        }
        if (score > bestScore && score > diag * 0.22) {
            float smin = 1e9f, smax = -1e9f;
            for (const auto &q : pts) {
                float s = (q - ls[i].a).dot(ls[i].dir);
                smin = std::min(smin, s);
                smax = std::max(smax, s);
            }
            bA = ls[i].a + ls[i].dir * smin;
            bB = ls[i].a + ls[i].dir * smax;
            bestScore = score; found = true;
        }
    }
    if (!found) return false;

    C.found = true; C.byColor = false;
    C.tip = bA; C.butt = bB; C.len = norm(bB - bA);
    return true;
}

/* ---- aim solver: cue axis -> ghost ball -> first object ball ------------ */
void PoolEngine::solveAim(Analysis &A, const Params &p)
{
    using namespace cv;
    AimSolution s;
    A.cueBallIdx = -1;
    if (!A.cue.found) { A.aim = s; return; }

    Point2f dir = A.cue.tip - A.cue.butt;
    double dl = norm(dir);
    if (dl < 1e-3) { A.aim = s; return; }
    dir *= (float)(1.0 / dl);
    s.dir = dir;
    s.tip = A.cue.tip;
    s.rayEnd = extendToRect(A.cue.tip, dir, A.frame.size());

    /* pick the cue ball: confirmed white track just ahead of the tip */
    int ci = -1; float bestPerp = 1e9f;
    for (size_t i = 0; i < A.tracks.size(); ++i) {
        const BallTrack &t = A.tracks[i];
        if (!t.isCue || !t.confirmed()) continue;
        Point2f rel = t.pos - A.cue.tip;
        float along = rel.dot(dir);
        float perp  = std::fabs(rel.cross(dir));
        if (along < -t.radius) continue;
        if (perp < bestPerp) { bestPerp = perp; ci = (int)i; }
    }

    float sumR = 0; int nr = 0;
    for (const auto &t : A.tracks) if (t.confirmed()) { sumR += t.radius; ++nr; }
    float avgR = nr ? sumR / nr : p.minBallR * 1.4f;
    float rc = avgR;
    if (ci >= 0) {
        s.haveCueBall = true;
        s.cueBallPos  = A.tracks[ci].pos;
        rc = A.tracks[ci].radius;
        A.cueBallIdx = ci;
    }
    s.ghostR = rc;

    /* ray-circle test: find first object ball the cue ball can hit */
    Point2f origin = s.haveCueBall ? s.cueBallPos : A.cue.tip;
    float bestT = 1e9f; int target = -1;
    for (size_t i = 0; i < A.tracks.size(); ++i) {
        if ((int)i == ci) continue;
        const BallTrack &t = A.tracks[i];
        if (!t.confirmed()) continue;
        float R = rc + t.radius;
        Point2f L = t.pos - origin;
        float tca = L.dot(dir);
        if (tca <= 0) continue;
        float d2 = L.dot(L) - tca * tca;
        if (d2 >= R * R) continue;
        float ti = tca - std::sqrt(R * R - d2); /// renamed to ti instead of t
        if (ti < 0) ti = 0;
        if (ti < bestT) { bestT = ti; target = (int)i; }
    }

    if (target >= 0) {
        s.valid = true;
        s.targetIdx = target;
        s.ghost = origin + dir * bestT;                 // ghost-ball centre
        const BallTrack &tb = A.tracks[target];
        Point2f od = tb.pos - s.ghost;
        double ol = norm(od);
        s.objDir = od * (float)(1.0 / std::max(ol, 1e-4));
        float cdot = std::max(-1.f, std::min(1.f, dir.dot(s.objDir)));
        s.cutDeg = std::acos(cdot) * 180.f / (float)CV_PI;
        Point2f defl = dir - s.objDir * cdot;           // cue tangent path
        double m = norm(defl);
        s.deflMag = (float)m;
        s.cueDir = (m > 0.06) ? defl * (float)(1.0 / m) : Point2f(0, 0);
    }
    A.aim = s;
}

/* ========================================================================== *
 *  Overlay drawing
 * ========================================================================== */

static cv::Mat drawOverlay(const Analysis &A, const Params &p,
                           double fps, bool rec)
{
    using namespace cv;
    Mat f = A.frame.clone();
    if (f.empty()) return f;

    /* table outline */
    if (p.showTable && A.tableContour.size() > 3)
        polylines(f, std::vector<std::vector<Point>>{A.tableContour},
                  true, Scalar(140, 110, 40), 1, LINE_AA);

    /* aim graphics */
    if (p.showAim && A.cue.found) {
        const AimSolution &s = A.aim;
        if (s.valid && s.targetIdx >= 0 && s.targetIdx < (int)A.tracks.size()) {
            const BallTrack &tb = A.tracks[s.targetIdx];

            if (s.haveCueBall)   // cue ball travel path
                line(f, s.cueBallPos, s.ghost, Scalar(255, 220, 80), 2, LINE_AA);
            dashedCircle(f, s.ghost, s.ghostR, Scalar(255, 255, 255), 1); // ghost
            circle(f, s.ghost, 2, Scalar(255, 255, 255), -1);

            arrowedLine(f, tb.pos, tb.pos + s.objDir * (tb.radius * 9.f),
                        Scalar(255, 60, 220), 2, LINE_AA, 0, 0.10); // object path
            if (s.deflMag > 0.06f)
                arrowedLine(f, s.ghost,
                            s.ghost + s.cueDir * (tb.radius * (3.f + 6.f * s.deflMag)),
                            Scalar(60, 160, 255), 2, LINE_AA, 0, 0.12); // cue deflect

            circle(f, tb.pos, (int)(tb.radius + 3), Scalar(255, 60, 220), 2, LINE_AA);
            putText(f, "TARGET", tb.pos + Point2f(tb.radius + 6, -tb.radius - 4),
                    FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255, 60, 220), 1, LINE_AA);
            char cb[48];
            snprintf(cb, sizeof(cb), "cut %.0f deg", s.cutDeg);
            putText(f, cb, s.ghost + Point2f(8, 16),
                    FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255, 255, 255), 1, LINE_AA);
        } else {
            dashedLine(f, s.tip, s.rayEnd, Scalar(120, 200, 255), 1);
            putText(f, "no ball on aim line", s.tip + Point2f(10, -8),
                    FONT_HERSHEY_SIMPLEX, 0.45, Scalar(120, 200, 255), 1, LINE_AA);
        }
        if (!s.haveCueBall)
            putText(f, "cue ball?", A.cue.tip + Point2f(-40, -12),
                    FONT_HERSHEY_SIMPLEX, 0.45, Scalar(80, 200, 255), 1, LINE_AA);
    }

    /* tracked balls */
    if (p.showBalls) {
        for (const auto &t : A.tracks) {
            if (!t.confirmed()) continue;
            circle(f, t.pos, (int)t.radius, t.bgr, 2, LINE_AA);
            circle(f, t.pos, 2, Scalar(0, 0, 0), -1);
            char lb[48];
            snprintf(lb, sizeof(lb), "#%d %s", t.id, t.label.c_str());
            putText(f, lb, t.pos + Point2f(-t.radius, -(t.radius + 5)),
                    FONT_HERSHEY_SIMPLEX, 0.42, Scalar(255, 255, 255), 1, LINE_AA);
            if (t.isCue)
                circle(f, t.pos, (int)(t.radius + 3), Scalar(80, 220, 255), 1, LINE_AA);
        }
    }

    /* cue stick */
    if (p.showCue && A.cue.found) {
        line(f, A.cue.butt, A.cue.tip, Scalar(40, 200, 255), 3, LINE_AA);
        circle(f, A.cue.tip, 4, Scalar(0, 255, 255), -1, LINE_AA);
    }

    /* HUD bar */
    if (f.rows >= 28) {
        Mat roi = f(Rect(0, 0, f.cols, 28));
        roi = roi * 0.45;
        int nB = 0;
        for (const auto &t : A.tracks) if (t.confirmed()) ++nB;
        std::string aim;
        if (!A.cue.found)        aim = "cue: --";
        else if (!A.aim.valid)   aim = "cue: OK | no target";
        else aim = "cue: OK | target #" +
                   std::to_string(A.tracks[A.aim.targetIdx].id) + " " +
                   A.tracks[A.aim.targetIdx].label;
        char hb[256];
        snprintf(hb, sizeof(hb), "FPS %4.1f | balls %d | %s | %s", fps, nB,
                 A.cue.found ? (A.cue.byColor ? "cue[colour]" : "cue[shape]") : "",
                 aim.c_str());
        putText(f, hb, Point(8, 19), FONT_HERSHEY_SIMPLEX, 0.5,
                Scalar(90, 255, 120), 1, LINE_AA);
        if (rec)
            putText(f, "REC", Point(f.cols - 46, 19),
                    FONT_HERSHEY_SIMPLEX, 0.55, Scalar(60, 60, 255), 2, LINE_AA);
    }
    return f;
}

/* ========================================================================== *
 *  Grabber — camera capture on its own thread
 * ========================================================================== */

class Grabber : public QThread
{
public:
    explicit Grabber(int index) : idx(index) {}
    ~Grabber() override { stop(); wait(1500); }

    void stop() { runFlag = false; }
    bool isOpen() const { return okFlag; }

    bool latest(cv::Mat &dst)
    {
        QMutexLocker lk(&mtx);
        if (frame.empty()) return false;
        frame.copyTo(dst);
        return true;
    }

protected:
    void run() override
    {
        cv::VideoCapture cap;
        cap.open(idx);
        if (!cap.isOpened()) { okFlag = false; return; }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        okFlag = true;

        cv::Mat tmp;
        while (runFlag) {
            if (!cap.read(tmp) || tmp.empty()) { msleep(40); continue; }
            {
                QMutexLocker lk(&mtx);
                tmp.copyTo(frame);
            }
        }
        cap.release();
    }

private:
    int idx;
    QMutex mtx;
    cv::Mat frame;
    std::atomic<bool> runFlag{true};
    std::atomic<bool> okFlag{false};
};

/* ========================================================================== *
 *  MainWindow — Qt GUI
 * ========================================================================== */

class MainWindow : public QWidget
{
public:
    explicit MainWindow(QWidget *parent = nullptr);

protected:
    bool eventFilter(QObject *, QEvent *) override;
    void resizeEvent(QResizeEvent *) override;
    void closeEvent(QCloseEvent *) override;

private:
    QSlider *addSlider(QGridLayout *g, int row, const QString &name,
                       int mn, int mx, int val);
    Params gatherParams() const;
    void onTick();
    void openCamera();
    void closeCamera();
    void closeCameraSoft();
    void toggleRecord(bool on);
    void stopRecording();
    void fitToView();
    void updateStatus(const Analysis &A);
    cv::Point2f mapToImage(const QPoint &lp) const;

    QLabel      *view = nullptr, *statusLbl = nullptr;
    QSpinBox    *camIdx = nullptr;
    QCheckBox   *flipChk = nullptr, *fallbackChk = nullptr;
    QCheckBox   *showBallsChk = nullptr, *showCueChk = nullptr;
    QCheckBox   *showAimChk = nullptr,  *showTableChk = nullptr;
    QPushButton *openBtn = nullptr, *closeBtn = nullptr, *recordBtn = nullptr;
    QPushButton *pickBtn = nullptr, *resetBtn = nullptr;
    QComboBox   *cueModeBox = nullptr;
    QSlider     *hueCSlider = nullptr, *hueWSlider = nullptr;
    QSlider     *sMinSlider = nullptr, *vMinSlider = nullptr, *minRSlider = nullptr;
    QTimer      *timer = nullptr;
    Grabber     *grab = nullptr;
    PoolEngine   engine;
    cv::VideoWriter writer;
    bool recording = false, picking = false, camAlive = false;
    QImage  lastImg;
    cv::Mat lastBgr;
    double  fpsEma = 0;
    QElapsedTimer fpsTimer, openElapsed;
};

MainWindow::MainWindow(QWidget *parent) : QWidget(parent)
{
    setWindowTitle("PoolShark — AR pool shot assistant (demo)");

    /* ---------------- side panel ---------------- */
    QWidget *side = new QWidget;
    side->setFixedWidth(330);
    QVBoxLayout *sv = new QVBoxLayout(side);
    sv->setContentsMargins(0, 0, 6, 0);

    QGroupBox *gCam = new QGroupBox("Camera (recording glasses)");
    QGridLayout *camL = new QGridLayout(gCam);
    camL->addWidget(new QLabel("device index"), 0, 0);
    camIdx = new QSpinBox;
    camIdx->setRange(0, 16);
    camL->addWidget(camIdx, 0, 1);
    flipChk = new QCheckBox("mirror horizontally");
    camL->addWidget(flipChk, 1, 0, 1, 2);
    openBtn  = new QPushButton("Open camera");
    closeBtn = new QPushButton("Close");
    closeBtn->setEnabled(false);
    camL->addWidget(openBtn, 2, 0);
    camL->addWidget(closeBtn, 2, 1);
    recordBtn = new QPushButton("\u25CF  record overlay video");
    recordBtn->setCheckable(true);
    recordBtn->setEnabled(false);
    camL->addWidget(recordBtn, 3, 0, 1, 2);
    sv->addWidget(gCam);

    QGroupBox *gCue = new QGroupBox("Cue-stick detection");
    QGridLayout *cueL = new QGridLayout(gCue);
    cueL->addWidget(new QLabel("mode"), 0, 0);
    cueModeBox = new QComboBox;
    cueModeBox->addItem("by colour");
    cueModeBox->addItem("by shape (any colour)");
    cueL->addWidget(cueModeBox, 0, 1, 1, 2);
    fallbackChk = new QCheckBox("fall back to shape");
    fallbackChk->setChecked(true);
    cueL->addWidget(fallbackChk, 1, 0, 1, 3);
    hueCSlider = addSlider(cueL, 2, "hue centre",     0, 179, 22);
    hueWSlider = addSlider(cueL, 3, "hue window",     1,  89, 16);
    sMinSlider = addSlider(cueL, 4, "min saturation", 0, 255, 40);
    vMinSlider = addSlider(cueL, 5, "min value",      0, 255, 70);
    pickBtn = new QPushButton("pick cue colour from image\u2026");
    cueL->addWidget(pickBtn, 6, 0, 1, 3);
    sv->addWidget(gCue);

    QGroupBox *gBall = new QGroupBox("Balls / tracking");
    QGridLayout *ballL = new QGridLayout(gBall);
    minRSlider = addSlider(ballL, 0, "min ball radius (px)", 4, 40, 9);
    resetBtn = new QPushButton("reset tracker");
    ballL->addWidget(resetBtn, 1, 0, 1, 3);
    sv->addWidget(gBall);

    QGroupBox *gOv = new QGroupBox("Overlay");
    QVBoxLayout *ovL = new QVBoxLayout(gOv);
    showBallsChk = new QCheckBox("balls + IDs");         showBallsChk->setChecked(true);
    showCueChk   = new QCheckBox("cue stick");           showCueChk->setChecked(true);
    showAimChk   = new QCheckBox("aim / ghost / paths"); showAimChk->setChecked(true);
    showTableChk = new QCheckBox("table outline");       showTableChk->setChecked(true);
    ovL->addWidget(showBallsChk);
    ovL->addWidget(showCueChk);
    ovL->addWidget(showAimChk);
    ovL->addWidget(showTableChk);
    sv->addWidget(gOv);

    QLabel *help = new QLabel(
        "1. open the glasses camera\n"
        "2. press 'pick cue colour' and click the stick\n"
        "   (or switch to shape mode for any colour)\n"
        "3. overlay: cue -> cue ball -> ghost ball ->\n"
        "   first object ball (magenta = object path,\n"
        "   orange = cue deflection after contact)");
    help->setStyleSheet("color:#666;");
    sv->addWidget(help);
    sv->addStretch(1);

    /* ---------------- video view ---------------- */
    view = new QLabel;
    view->setMinimumSize(640, 480);
    view->setAlignment(Qt::AlignCenter);
    view->setStyleSheet("background-color:#101014;color:#777;border:1px solid #333;");
    view->setText("no camera — set a device index and press 'Open camera'");
    view->installEventFilter(this);

    QHBoxLayout *split = new QHBoxLayout;
    split->addWidget(side);
    split->addWidget(view, 1);

    statusLbl = new QLabel("idle");
    statusLbl->setStyleSheet("color:#9c9;");

    QVBoxLayout *outer = new QVBoxLayout(this);
    outer->addLayout(split, 1);
    outer->addWidget(statusLbl);

    connect(openBtn,   &QPushButton::clicked, this, &MainWindow::openCamera);
    connect(closeBtn,  &QPushButton::clicked, this, &MainWindow::closeCamera);
    connect(recordBtn, &QPushButton::toggled, this, &MainWindow::toggleRecord);
    connect(resetBtn,  &QPushButton::clicked, this, [this] { engine.reset(); });
    connect(pickBtn,   &QPushButton::clicked, this, [this] {
        picking = !picking;
        pickBtn->setText(picking ? "now click the cue stick\u2026"
                                 : "pick cue colour from image\u2026");
        if (picking)
            statusLbl->setText("click on the cue stick in the video to sample its colour");
    });

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::onTick);
    timer->start(33);
}

/* ---- UI helpers ---------------------------------------------------------- */

QSlider *MainWindow::addSlider(QGridLayout *g, int row, const QString &name,
                               int mn, int mx, int val)
{
    g->addWidget(new QLabel(name), row, 0);
    QSlider *s = new QSlider(Qt::Horizontal);
    s->setRange(mn, mx);
    s->setValue(val);
    QLabel *v = new QLabel(QString::number(val));
    v->setMinimumWidth(36);
    g->addWidget(s, row, 1);
    g->addWidget(v, row, 2);
    QObject::connect(s, QOverload<int>::of(&QSlider::valueChanged),
                     v, [v](int x) { v->setText(QString::number(x)); });
    return s;
}

Params MainWindow::gatherParams() const
{
    Params p;
    p.flip          = flipChk->isChecked();
    p.cueMode       = cueModeBox->currentIndex();
    p.shapeFallback = fallbackChk->isChecked();
    p.hueC = hueCSlider->value();
    p.hueW = hueWSlider->value();
    p.sMin = sMinSlider->value();
    p.vMin = vMinSlider->value();
    p.minBallR = (float)minRSlider->value();
    p.showBalls = showBallsChk->isChecked();
    p.showCue   = showCueChk->isChecked();
    p.showAim   = showAimChk->isChecked();
    p.showTable = showTableChk->isChecked();
    return p;
}

/* ---- camera control ------------------------------------------------------ */

void MainWindow::openCamera()
{
    closeCameraSoft();
    engine.reset();
    camAlive = false;
    grab = new Grabber(camIdx->value());
    grab->start();
    openElapsed.restart();
    closeBtn->setEnabled(true);
    statusLbl->setText(QString("opening camera %1\u2026").arg(camIdx->value()));
}

void MainWindow::closeCameraSoft()
{
    if (recordBtn->isChecked()) recordBtn->setChecked(false);
    stopRecording();
    if (grab) { grab->stop(); grab->wait(2000); delete grab; grab = nullptr; }
    recordBtn->setEnabled(false);
    camAlive = false;
}

void MainWindow::closeCamera()
{
    closeCameraSoft();
    view->clear();
    view->setText("camera closed");
    lastImg = QImage();
    lastBgr.release();
    closeBtn->setEnabled(false);
    statusLbl->setText("camera closed");
}

void MainWindow::toggleRecord(bool on)
{
    recording = on;
    recordBtn->setText(on ? "\u25A0  stop recording"
                          : "\u25CF  record overlay video");
    if (!on) stopRecording();
}

void MainWindow::stopRecording()
{
    if (writer.isOpened()) writer.release();
    recording = false;
}

/* ---- main processing tick ------------------------------------------------ */

void MainWindow::onTick()
{
    if (!grab) return;

    cv::Mat frame;
    if (!grab->latest(frame)) {
        if (!grab->isOpen() && openElapsed.elapsed() > 2500)
            statusLbl->setText("camera failed to open (wrong index / busy?)");
        return;
    }
    if (!camAlive) { camAlive = true; recordBtn->setEnabled(true); }

    double dt = fpsTimer.restart();
    fpsEma = fpsEma * 0.9 + (1000.0 / qMax(1.0, dt)) * 0.1;

    Params p = gatherParams();
    Analysis A;
    engine.process(frame, p, A);
    if (A.frame.empty()) return;

    lastBgr = A.frame.clone();                      // for colour picking
    cv::Mat shown = drawOverlay(A, p, fpsEma, recording);

    if (recording) {
        if (!writer.isOpened()) {
            QString fn = QDateTime::currentDateTime()
                             .toString("'poolshark_'yyyyMMdd_HHmmss'.avi'");
            writer.open(fn.toStdString(),
                        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                        30.0, shown.size());
            if (!writer.isOpened()) {
                statusLbl->setText("could not start video writer");
                recording = false;
                recordBtn->setChecked(false);
            }
        }
        if (writer.isOpened()) writer.write(shown);
    }

    lastImg = matToQImage(shown);
    fitToView();
    updateStatus(A);
}

void MainWindow::updateStatus(const Analysis &A)
{
    int n = 0;
    for (const auto &t : A.tracks) if (t.confirmed()) ++n;

    QString cueTxt = !A.cue.found ? "cue: not found"
                   : A.cue.byColor ? "cue: colour" : "cue: shape";
    QString aimTxt;
    if (A.cue.found) {
        if (A.aim.valid) {
            aimTxt = QString("target %1#%2 | cut %3\u00B0%4")
                .arg(QString::fromStdString(A.tracks[A.aim.targetIdx].label))
                .arg(A.tracks[A.aim.targetIdx].id)
                .arg(A.aim.cutDeg, 0, 'f', 0)
                .arg(A.aim.haveCueBall ? "" : " | cue ball NOT locked");
        } else {
            aimTxt = "aim: no ball on line";
        }
    }
    statusLbl->setText(QString("FPS %1 | balls %2 | %3 | %4%5")
        .arg(fpsEma, 0, 'f', 1).arg(n).arg(cueTxt).arg(aimTxt)
        .arg(recording ? " | \u25CF REC" : ""));
}

/* ---- colour picking straight from the video ------------------------------ */

bool MainWindow::eventFilter(QObject *o, QEvent *ev)
{
    if (o == view && picking && ev->type() == QEvent::MouseButtonPress) {
        QMouseEvent *me = static_cast<QMouseEvent *>(ev);
        cv::Point2f ip = mapToImage(me->pos());
        if (!lastBgr.empty() && ip.x >= 0 && ip.y >= 0 &&
            ip.x < lastBgr.cols - 1 && ip.y < lastBgr.rows - 1) {
            cv::Rect r((int)ip.x - 6, (int)ip.y - 6, 13, 13);
            r &= cv::Rect(0, 0, lastBgr.cols, lastBgr.rows);
            if (r.width >= 3 && r.height >= 3) {
                cv::Mat hsvs;
                cv::cvtColor(lastBgr(r), hsvs, cv::COLOR_BGR2HSV);
                cv::Scalar m = cv::mean(hsvs);
                hueCSlider->setValue((int)m[0]);
                hueWSlider->setValue(16);
                sMinSlider->setValue(qMax(5, (int)m[1] - 80));
                vMinSlider->setValue(qMax(5, (int)m[2] - 80));
                cueModeBox->setCurrentIndex(0);
                statusLbl->setText(QString("cue colour sampled: H%1 S%2 V%3")
                    .arg((int)m[0]).arg((int)m[1]).arg((int)m[2]));
            }
        }
        picking = false;
        pickBtn->setText("pick cue colour from image\u2026");
        return true;
    }
    return QWidget::eventFilter(o, ev);
}

cv::Point2f MainWindow::mapToImage(const QPoint &lp) const
{
    if (lastImg.isNull()) return cv::Point2f(-1, -1);
    double sx = (double)view->width()  / lastImg.width();
    double sy = (double)view->height() / lastImg.height();
    double s  = qMin(sx, sy);
    double ox = (view->width()  - lastImg.width()  * s) * 0.5;
    double oy = (view->height() - lastImg.height() * s) * 0.5;
    return cv::Point2f((float)((lp.x() - ox) / s), (float)((lp.y() - oy) / s));
}

/* ---- display ------------------------------------------------------------- */

void MainWindow::fitToView()
{
    if (lastImg.isNull()) return;
    view->setPixmap(QPixmap::fromImage(lastImg)
        .scaled(view->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void MainWindow::resizeEvent(QResizeEvent *e)
{
    QWidget::resizeEvent(e);
    fitToView();
}

void MainWindow::closeEvent(QCloseEvent *e)
{
    timer->stop();
    stopRecording();
    if (grab) { grab->stop(); grab->wait(2000); }
    QWidget::closeEvent(e);
}

/* ========================================================================== */

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    app.setApplicationName("PoolShark");
    app.setStyle("Fusion");

    MainWindow w;
    w.resize(1280, 780);
    w.show();
    return app.exec();
}
