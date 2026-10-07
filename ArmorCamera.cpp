//-------------------------------------------------------------
/**
\file      GxSingleCamColor.cpp
\brief     Sample to show how to acquire color image continuously and save ppm images
\version   1.1.2504.9221
\date      2025.4.22
*/
//-------------------------------------------------------------

#include <GxIAPI.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include "Common/Common.h"

#include <opencv2/opencv.hpp>
#include "HybridArmorDetector.h"
#include <spdlog/spdlog.h>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace cv;
using namespace std;


#define ACQ_BUFFER_NUM 5              ///< Acquisition Buffer Qty.
#define ACQ_TRANSFER_SIZE (64 * 1024) ///< Size of data transfer block
#define ACQ_TRANSFER_NUMBER_URB 64    ///< Qty. of data transfer block
#define FILE_NAME_LEN 50              ///< Save image file name length

#define PIXFMT_CVT_FAIL -1   ///< PixelFormatConvert fail
#define PIXFMT_CVT_SUCCESS 0 ///< PixelFormatConvert success

// Show error message
#define GX_VERIFY(emStatus)            \
    if (emStatus != GX_STATUS_SUCCESS) \
    {                                  \
        GetErrorString(emStatus);      \
        return emStatus;               \
    }

// Show error message, close device and lib
#define GX_VERIFY_EXIT(emStatus)       \
    if (emStatus != GX_STATUS_SUCCESS) \
    {                                  \
        GetErrorString(emStatus);      \
        GXCloseDevice(g_hDevice);      \
        g_hDevice = NULL;              \
        GXCloseLib();                  \
        printf("<App Exit!>\n");       \
        return emStatus;               \
    }

GX_DEV_HANDLE g_hDevice = NULL;                  ///< Device handle
bool g_bAcquisitionFlag = false;                 ///< Thread running flag
bool g_bSavePPMImage = false;                    ///< Save raw image flag
pthread_t g_nAcquisitonThreadID = 0;             ///< Thread ID of Acquisition thread

unsigned char *g_pDesImageBuf = NULL; ///< Memory for RGB24
uint32_t g_nDesImageLen = NULL; ///< Memory length for RGB24

uint32_t g_nPayloadSize = 0; ///< Payload size

// Allocate the memory for pixel format transform
void PreForAcquisition();

// Release the memory allocated
void UnPreForAcquisition();

// Convert frame date to suitable pixel format
int PixelFormatConvert(PGX_FRAME_BUFFER);

// Save one frame to PPM image file
void SavePPMFile(uint32_t, uint32_t);

// Acquisition thread function
void *ProcGetImage(void *);

// Get description of error
void GetErrorString(GX_STATUS);

int main()
{
    printf("\n");
    printf("-------------------------------------------------------------\n");
    printf("Sample to show how to acquire color image continuously and save ppm image.\n");
    printf("version: 1.1.2604.9011\n");
    printf("-------------------------------------------------------------\n");
    printf("\n");
    printf("Initializing......");
    printf("\n\n");

    GX_STATUS emStatus = GX_STATUS_SUCCESS;

    uint32_t ui32DeviceNum = 0;

    // Initialize libary
    emStatus = GXInitLib();
    if (emStatus != GX_STATUS_SUCCESS)
    {
        GetErrorString(emStatus);
        return emStatus;
    }

    // Get device enumerated number
    emStatus = GXUpdateAllDeviceList(&ui32DeviceNum, 1000);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        GetErrorString(emStatus);
        GXCloseLib();
        return emStatus;
    }

    // If no device found, app exit
    if (ui32DeviceNum <= 0)
    {
        printf("<No device found>\n");
        GXCloseLib();
        return emStatus;
    }

    // Open first device enumerated
    emStatus = GXOpenDeviceByIndex(1, &g_hDevice);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        GetErrorString(emStatus);
        GXCloseLib();
        return emStatus;
    }

    // Get Device Info
    printf("***********************************************\n");
    // Get libary version
    printf("<Libary Version : %s>\n", GXGetLibVersion());

    // Get string length of Vendor name
    GX_STRING_VALUE stStrVendorName;
    emStatus = GXGetStringValue(g_hDevice, "DeviceVendorName", &stStrVendorName);
    GX_VERIFY_EXIT(emStatus);
    printf("<Vendor Name : %s>\n", stStrVendorName.strCurValue);

    // Get string length of Model name
    GX_STRING_VALUE stStrModelName;
    emStatus = GXGetStringValue(g_hDevice, "DeviceModelName", &stStrModelName);
    GX_VERIFY_EXIT(emStatus);
    printf("<Model Name : %s>\n", stStrModelName.strCurValue);

    // Get string length of Serial number
    GX_STRING_VALUE stStrSerialNumber;
    emStatus = GXGetStringValue(g_hDevice, "DeviceSerialNumber", &stStrSerialNumber);
    GX_VERIFY_EXIT(emStatus);
    printf("<Serial Number : %s>\n", stStrSerialNumber.strCurValue);

    // Get string length of Device version
    GX_STRING_VALUE stStrDeviceVersion;
    emStatus = GXGetStringValue(g_hDevice, "DeviceVersion", &stStrDeviceVersion);
    GX_VERIFY_EXIT(emStatus);
    printf("<Device Version : %s>\n", stStrDeviceVersion.strCurValue);

    printf("***********************************************\n");

    // Get the type of pixel format. whether is a color pixel format.
    GX_ENUM_VALUE emValue;
    emStatus = GXGetEnumValue(g_hDevice, "PixelFormat", &emValue);
    GX_VERIFY_EXIT(emStatus);

    bool bColorPixelFormat = !IsGrayPixelFormat((GX_PIXEL_FORMAT_ENTRY)emValue.stCurValue.nCurValue);
    // This app only support color cameras
    if (!bColorPixelFormat)
    {
        printf("<This app only support color cameras! App Exit!>\n");
        GXCloseDevice(g_hDevice);
        g_hDevice = NULL;
        GXCloseLib();
        return 0;
    }

    uint32_t nDSNum = 0;
    emStatus = GXGetDataStreamNumFromDev(g_hDevice, &nDSNum);
    GX_VERIFY_EXIT(emStatus);

    if (nDSNum < 1)
    {
        printf("Failed to obtain the number of streams\n");
        return GX_STATUS_ERROR;
    }

    GX_DS_HANDLE phDS = NULL;
    emStatus = GXGetDataStreamHandleFromDev(g_hDevice, 1, &phDS);
    GX_VERIFY_EXIT(emStatus);
    if (NULL == phDS)
    {
        printf("Failed to get data stream handle\n");
        return GX_STATUS_INVALID_HANDLE;
    }

    emStatus = GXGetPayLoadSize(phDS, &g_nPayloadSize);
    GX_VERIFY_EXIT(emStatus);

    printf("\n");
    printf("Press [a] or [A] and then press [Enter] to start acquisition\n");
    printf("Press [s] or [S] and then press [Enter] to save one ppm image\n");
    printf("Press [x] or [X] and then press [Enter] to Exit the Program\n");
    printf("\n");

    int nStartKey = 0;
    bool bWaitStart = true;
    while (bWaitStart)
    {
        nStartKey = getchar();
        switch (nStartKey)
        {
        // press 'a' and [Enter] to start acquisition
        // press 'x' and [Enter] to exit
        case 'a':
        case 'A':
            // Start to acquisition
            bWaitStart = false;
            break;
        case 'S':
        case 's':
            printf("<Please start acquisiton before saving image!>\n");
            break;
        case 'x':
        case 'X':
            // App exit
            GXCloseDevice(g_hDevice);
            g_hDevice = NULL;
            GXCloseLib();
            printf("<App exit!>\n");
            return 0;
        default:
            break;
        }
    }

    // Set PacketSize
    GX_NODE_ACCESS_MODE emAccessMode;
    emStatus = GXGetNodeAccessMode(g_hDevice, "GevSCPSPacketSize", &emAccessMode);
    bool bImplementPacketSize = (emAccessMode == GX_NODE_ACCESS_MODE_RW) ? true : false;
    if (bImplementPacketSize)
    {
        uint32_t ui32PacketSize = 0;
        emStatus = GXGetOptimalPacketSize(g_hDevice, &ui32PacketSize);
        if (emStatus != GX_STATUS_SUCCESS)
        {
            printf("<Failed to get OptimalPacketSize!>");
        }
        emStatus = GXSetIntValue(g_hDevice, "GevSCPSPacketSize", ui32PacketSize);
        if (emStatus != GX_STATUS_SUCCESS)
        {
            printf("<Failed to set GevSCPSPacketSize!>");
        }
    }

    // Set PacketDelay
    emStatus = GXGetNodeAccessMode(g_hDevice, "GevSCPD", &emAccessMode);
    bool bImplementPacketDelay = (emAccessMode == GX_NODE_ACCESS_MODE_RW) ? true : false;
    if (bImplementPacketDelay)
    {
        GX_INT_VALUE stValue;
        emStatus = GXGetIntValue(g_hDevice, "GevSCPD", &stValue);
        if (emStatus != GX_STATUS_SUCCESS)
        {
            printf("<Failed to get GevSCPD!>");
        }
        emStatus = GXSetIntValue(g_hDevice, "GevSCPD", stValue.nMin);
        if (emStatus != GX_STATUS_SUCCESS)
        {
            printf("<Failed to set GevSCPD!>");
        }
    }

    // Set acquisition mode
    emStatus = GXSetEnumValueByString(g_hDevice, "AcquisitionMode", "Continuous");
    GX_VERIFY_EXIT(emStatus);

    // Set trigger mode
    emStatus = GXSetEnumValueByString(g_hDevice, "TriggerMode", "Off");
    GX_VERIFY_EXIT(emStatus);

    // Set buffer quantity of acquisition queue
    uint64_t nBufferNum = ACQ_BUFFER_NUM;
    emStatus = GXSetAcqusitionBufferNumber(g_hDevice, nBufferNum);
    GX_VERIFY_EXIT(emStatus);

    GX_NODE_ACCESS_MODE emStreamTransferSizeIm;
    emStatus = GXGetNodeAccessMode(phDS, "StreamTransferSize", &emStreamTransferSizeIm);
    GX_VERIFY_EXIT(emStatus);
    bool bStreamTransferSize = ((emStreamTransferSizeIm == GX_NODE_ACCESS_MODE_RO) || (emStreamTransferSizeIm == GX_NODE_ACCESS_MODE_WO) || (emStreamTransferSizeIm == GX_NODE_ACCESS_MODE_RW)) ? true : false;

    if (bStreamTransferSize)
    {
        // Set size of data transfer block
        emStatus = GXSetIntValue(phDS, "StreamTransferSize", ACQ_TRANSFER_SIZE);
        GX_VERIFY_EXIT(emStatus);
    }

    GX_NODE_ACCESS_MODE emStreamTransferUrbIm;
    emStatus = GXGetNodeAccessMode(phDS, "StreamTransferNumberUrb", &emStreamTransferUrbIm);
    GX_VERIFY_EXIT(emStatus);
    bool bStreamTransferNumberUrb = ((emStreamTransferUrbIm == GX_NODE_ACCESS_MODE_RO) || (emStreamTransferUrbIm == GX_NODE_ACCESS_MODE_WO) || (emStreamTransferUrbIm == GX_NODE_ACCESS_MODE_RW)) ? true : false;

    if (bStreamTransferNumberUrb)
    {
        // Set qty. of data transfer block
        emStatus = GXSetIntValue(phDS, "StreamTransferNumberUrb", ACQ_TRANSFER_NUMBER_URB);
        GX_VERIFY_EXIT(emStatus);
    }

    // Set Balance White Mode : Continuous
    GX_NODE_ACCESS_MODE emBalanceWhiteAutoAccess;
    emStatus = GXGetNodeAccessMode(phDS, "BalanceWhiteAuto", &emBalanceWhiteAutoAccess);
    GX_VERIFY_EXIT(emStatus);
    if( GX_NODE_ACCESS_MODE_RW == emBalanceWhiteAutoAccess)
    {
        emStatus = GXSetEnumValueByString(g_hDevice, "BalanceWhiteAuto", "Once");
        GX_VERIFY_EXIT(emStatus);
    }

    // Allocate the memory for pixel format transform
    PreForAcquisition();

    // Device start acquisition
    emStatus = GXStreamOn(g_hDevice);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        // Release the memory allocated
        UnPreForAcquisition();
        GX_VERIFY_EXIT(emStatus);
    }

    // Start acquisition thread, if thread create failed, exit this app
    int nRet = pthread_create(&g_nAcquisitonThreadID, NULL, ProcGetImage, NULL);
    if (nRet != 0)
    {
        // Release the memory allocated
        UnPreForAcquisition();

        GXCloseDevice(g_hDevice);
        g_hDevice = NULL;
        GXCloseLib();

        printf("<Failed to create the acquisition thread, App Exit!>\n");
        exit(nRet);
    }

    // Main loop
    bool bRun = true;
    while (bRun == true)
    {
        int nKey = getchar();
        // press 's' and [Enter] to save image
        // press 'x' and [Enter] to exit
        switch (nKey)
        {
        // Save PPM Image
        case 'S':
        case 's':
            g_bSavePPMImage = true;
            break;
        // Exit app
        case 'X':
        case 'x':
            bRun = false;
            break;
        default:
            break;
        }
    }

    // Stop Acquisition thread
    g_bAcquisitionFlag = false;
    pthread_join(g_nAcquisitonThreadID, NULL);

    // Device stop acquisition
    emStatus = GXStreamOff(g_hDevice);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        // Release the memory allocated
        UnPreForAcquisition();
        GX_VERIFY_EXIT(emStatus);
    }

    // Release the resources and stop acquisition thread
    UnPreForAcquisition();

    // Close device
    emStatus = GXCloseDevice(g_hDevice);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        GetErrorString(emStatus);
        g_hDevice = NULL;
        GXCloseLib();
        return emStatus;
    }

    // Release libary
    emStatus = GXCloseLib();
    if (emStatus != GX_STATUS_SUCCESS)
    {
        GetErrorString(emStatus);
        return emStatus;
    }

    printf("<App exit!>\n");
    return 0;
}

//-------------------------------------------------
/**
\brief Save PPM image
\param ui32Width[in]       image width
\param ui32Height[in]      image height
\return void
*/
//-------------------------------------------------
void SavePPMFile(uint32_t ui32Width, uint32_t ui32Height)
{
    char szName[FILE_NAME_LEN] = {0};

    static int nRawFileIndex = 0;

    if (g_pDesImageBuf != NULL)
    {
        FILE *phImageFile = NULL;
        snprintf(szName, FILE_NAME_LEN, "Frame_%d.ppm", nRawFileIndex++);
        phImageFile = fopen(szName, "wb");
        if (phImageFile == NULL)
        {
            printf("Create or Open %s failed!\n", szName);
            return;
        }
        // Save color image
        fprintf(phImageFile, "P6\n%u %u 255\n", ui32Width, ui32Height);
        fwrite(g_pDesImageBuf, 1, g_nDesImageLen, phImageFile);
        fclose(phImageFile);
        phImageFile = NULL;
        printf("Save %s succeed!\n", szName);
    }
    else
    {
        printf("Save %s failed!\n", szName);
    }
}

//-------------------------------------------------
/**
\brief Allocate the memory for pixel format transform
\return void
*/
//-------------------------------------------------
void PreForAcquisition()
{
    if( NULL == g_pDesImageBuf)
    {
        g_nDesImageLen = g_nPayloadSize * 3;
        g_pDesImageBuf = new unsigned char[g_nDesImageLen];
    }

    return;
}

//-------------------------------------------------
/**
\brief Release the memory allocated
\return void
*/
//-------------------------------------------------
void UnPreForAcquisition()
{
    // Release resources
    if (NULL != g_pDesImageBuf)
    {
        delete[] g_pDesImageBuf;
        g_pDesImageBuf = NULL;
    }

    return;
}

//-------------------------------------------------
/**
\brief Acquisition thread function
\param pParam       thread param, not used in this app
\return void*
*/
//-------------------------------------------------

// ============================================================
// RoboMaster armor detection
// Input: OpenCV BGR image
// ============================================================
void DetectArmor(cv::Mat& frame)
{
    static auto lastTime = std::chrono::steady_clock::now();

    // ---------- FPS ----------
    auto now = std::chrono::steady_clock::now();
    double dt =
        std::chrono::duration<double>(now - lastTime).count();
    lastTime = now;

    double fps = (dt > 0.000001) ? 1.0 / dt : 0.0;

    // ---------- BGR -> HSV ----------
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    // ---------- Red extraction ----------
    Mat mask1, mask2, redMask;

    inRange(
        hsv,
        Scalar(0, 150, 150),
        Scalar(10, 255, 255),
        mask1
    );

    inRange(
        hsv,
        Scalar(170, 150, 150),
        Scalar(180, 255, 255),
        mask2
    );

    redMask = mask1 | mask2;

    Mat kernel =
        getStructuringElement(MORPH_RECT, Size(3, 3));

    morphologyEx(
        redMask,
        redMask,
        MORPH_OPEN,
        kernel
    );

    // ---------- Find contours ----------
    vector<vector<Point>> contours;

    findContours(
        redMask,
        contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE
    );

    vector<RotatedRect> lights;

    for (const auto& contour : contours)
    {
        double area = contourArea(contour);

        // Real-time version uses a lower threshold than test.jpg.
        // Geometry filtering below performs the main rejection.
        if (area < 50)
            continue;

        RotatedRect rect = minAreaRect(contour);

        float width  = rect.size.width;
        float height = rect.size.height;

        if (width > height)
            swap(width, height);

        if (width <= 0)
            continue;

        float ratio = height / width;

        if (ratio > 2.0)
        {
            lights.push_back(rect);

            Point2f pts[4];
            rect.points(pts);

            for (int i = 0; i < 4; ++i)
            {
                line(
                    frame,
                    pts[i],
                    pts[(i + 1) % 4],
                    Scalar(0, 255, 0),
                    2
                );
            }
        }
    }

    // ---------- Find best light-bar pair ----------
    bool armorFound = false;

    RotatedRect bestLeft;
    RotatedRect bestRight;
    double bestScore = 1e18;

    for (size_t i = 0; i < lights.size(); ++i)
    {
        for (size_t j = i + 1; j < lights.size(); ++j)
        {
            RotatedRect a = lights[i];
            RotatedRect b = lights[j];

            if (a.center.x > b.center.x)
                swap(a, b);

            float ah = max(a.size.width, a.size.height);
            float bh = max(b.size.width, b.size.height);

            if (ah <= 0 || bh <= 0)
                continue;

            // Similar height
            float heightRatio =
                max(ah, bh) / min(ah, bh);

            if (heightRatio > 2.0)
                continue;

            // Vertical alignment
            float avgH = (ah + bh) * 0.5f;

            float dy =
                abs(a.center.y - b.center.y);

            if (dy > avgH)
                continue;

            // Horizontal separation
            float dx =
                abs(a.center.x - b.center.x);

            if (dx < avgH * 0.5f ||
                dx > avgH * 8.0f)
                continue;

            // Lower score = better pair
            double score =
                abs(ah - bh) +
                dy * 2.0;

            if (score < bestScore)
            {
                bestScore = score;
                bestLeft = a;
                bestRight = b;
                armorFound = true;
            }
        }
    }

    // ---------- Armor corners ----------
    if (armorFound)
    {
        Point2f leftPts[4];
        Point2f rightPts[4];

        bestLeft.points(leftPts);
        bestRight.points(rightPts);

        sort(
            leftPts,
            leftPts + 4,
            [](const Point2f& a, const Point2f& b)
            {
                return a.y < b.y;
            }
        );

        sort(
            rightPts,
            rightPts + 4,
            [](const Point2f& a, const Point2f& b)
            {
                return a.y < b.y;
            }
        );

        Point2f LT =
            (leftPts[0].x < leftPts[1].x)
            ? leftPts[0] : leftPts[1];

        Point2f LB =
            (leftPts[2].x < leftPts[3].x)
            ? leftPts[2] : leftPts[3];

        Point2f RT =
            (rightPts[0].x > rightPts[1].x)
            ? rightPts[0] : rightPts[1];

        Point2f RB =
            (rightPts[2].x > rightPts[3].x)
            ? rightPts[2] : rightPts[3];

        // Blue armor frame
        line(frame, LT, RT, Scalar(255, 0, 0), 3);
        line(frame, RT, RB, Scalar(255, 0, 0), 3);
        line(frame, RB, LB, Scalar(255, 0, 0), 3);
        line(frame, LB, LT, Scalar(255, 0, 0), 3);

        // Yellow corners
        circle(frame, LT, 6, Scalar(0, 255, 255), -1);
        circle(frame, LB, 6, Scalar(0, 255, 255), -1);
        circle(frame, RB, 6, Scalar(0, 255, 255), -1);
        circle(frame, RT, 6, Scalar(0, 255, 255), -1);

        putText(
            frame,
            "ARMOR",
            Point(
                max(0, (int)LT.x),
                max(30, (int)LT.y - 10)
            ),
            FONT_HERSHEY_SIMPLEX,
            0.8,
            Scalar(255, 0, 0),
            2
        );

        // Avoid flooding terminal every frame
        static int logCounter = 0;

        if (++logCounter >= 60)
        {
            spdlog::info(
                "Armor detected | candidates={} | FPS={:.1f}",
                lights.size(),
                fps
            );

            logCounter = 0;
        }
    }

    // ---------- FPS display ----------
    char fpsText[64];

    snprintf(
        fpsText,
        sizeof(fpsText),
        "FPS: %.1f",
        fps
    );

    putText(
        frame,
        fpsText,
        Point(20, 40),
        FONT_HERSHEY_SIMPLEX,
        1.0,
        Scalar(0, 255, 0),
        2
    );

    imshow("Armor Camera", frame);

    // ESC closes the OpenCV window
    waitKey(1);
}


void *ProcGetImage(void *pParam)
{
    GX_STATUS emStatus = GX_STATUS_SUCCESS;

    // Thread running flag setup
    g_bAcquisitionFlag = true;
    PGX_FRAME_DATA_EX pFrameBuffer = NULL;

    time_t lInit;
    time_t lEnd;
    uint32_t ui32FrameCount = 0;

    while (g_bAcquisitionFlag)
    {
        if (!ui32FrameCount)
        {
            time(&lInit);
        }

        // Get a frame from Queue
        emStatus = GXDQBufEx(g_hDevice, &pFrameBuffer, 1000);
        if (emStatus != GX_STATUS_SUCCESS)
        {
            if (emStatus == GX_STATUS_TIMEOUT)
            {
                continue;
            }
            else
            {
                GetErrorString(emStatus);
                break;
            }
        }

        if (pFrameBuffer->nStatus != GX_FRAME_STATUS_SUCCESS)
        {
            printf("<Abnormal Acquisition: Exception code: %d>\n", pFrameBuffer->nStatus);
        }
        else
        {
            ui32FrameCount++;
            time(&lEnd);
            // Print acquisition info each second.
            if (lEnd - lInit >= 1)
            {
                printf("<Successful acquisition: FrameCount: %u Width: %d Height: %d FrameID: %lu>\n",
                       ui32FrameCount, pFrameBuffer->nWidth, pFrameBuffer->nHeight, pFrameBuffer->nFrameID);
                ui32FrameCount = 0;
            }

        // Convert EVERY successfully acquired frame to RGB.
        g_nDesImageLen = g_nPayloadSize * 3;

        int nRet = PixelFormatConvert(
            pFrameBuffer,
            g_pDesImageBuf,
            &g_nDesImageLen,
            GX_PIXEL_FORMAT_RGB8
        );

        if (nRet == PIXFMT_CVT_SUCCESS)
        {
            cv::Mat rgb(
                pFrameBuffer->nHeight,
                pFrameBuffer->nWidth,
                CV_8UC3,
                g_pDesImageBuf
            );

            cv::Mat frame;
            cv::cvtColor(rgb, frame, cv::COLOR_RGB2BGR);
        // ====================================================
        // Final hybrid armor detection
        //
        // YOLO Pose first.
        // If YOLO completely misses -> OpenCV fallback.
        // Then RED/BLUE + number 1~5.
        //
        // static: models are loaded only once.
        // ====================================================
        static HybridArmorDetector detector(
            "models/armor_pose.onnx",
            "models/number_classifier.onnx"
        );

        static bool detector_status_logged = false;

        if (!detector_status_logged)
        {
            if (detector.isReady())
            {
                spdlog::info(
                    "Hybrid armor detector ready: "
                    "armor_pose.onnx + number_classifier.onnx"
                );
            }
            else
            {
                spdlog::error(
                    "Hybrid armor detector initialization failed"
                );
            }

            detector_status_logged = true;
        }

        auto detections =
            detector.detect(frame, true);

        // 不要每帧刷屏，每30帧输出一次调试信息
        if (ui32FrameCount % 30 == 0)
        {
            spdlog::info(
                "Armor detections: {}, pipeline FPS: {:.1f}, "
                "total: {:.1f} ms, pose: {:.1f} ms",
                detections.size(),
                detector.lastFps(),
                detector.lastTotalMs(),
                detector.lastPoseMs()
            );
        }

        cv::imshow("Armor Camera", frame);
        cv::waitKey(1);

            // Saving image remains optional.
            if (g_bSavePPMImage)
            {
                SavePPMFile(
                    pFrameBuffer->nWidth,
                    pFrameBuffer->nHeight
                );
                g_bSavePPMImage = false;
            }
        }
        else
        {
            printf("PixelFormat Convert failed!\n");
        }

    }

    emStatus = GXQBufEx(g_hDevice, pFrameBuffer);
        if (emStatus != GX_STATUS_SUCCESS)
        {
            GetErrorString(emStatus);
            break;
        }
    }
    printf("<Acquisition thread Exit!>\n");

    return 0;
}

//----------------------------------------------------------------------------------
/**
\brief  Get description of input error code
\param  emErrorStatus  error code

\return void
*/
//----------------------------------------------------------------------------------
void GetErrorString(GX_STATUS emErrorStatus)
{
    char *error_info = NULL;
    size_t size = 0;
    GX_STATUS emStatus = GX_STATUS_SUCCESS;

    // Get length of error description
    emStatus = GXGetLastError(&emErrorStatus, NULL, &size);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        printf("<Error when calling GXGetLastError>\n");
        return;
    }

    // Alloc error resources
    error_info = new char[size];
    if (error_info == NULL)
    {
        printf("<Failed to allocate memory>\n");
        return;
    }

    // Get error description
    emStatus = GXGetLastError(&emErrorStatus, error_info, &size);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        printf("<Error when calling GXGetLastError>\n");
    }
    else
    {
        printf("%s\n", error_info);
    }

    // Realease error resources
    if (error_info != NULL)
    {
        delete[] error_info;
        error_info = NULL;
    }
}
