//-------------------------------------------------------------
/**
\file      Common.h
\brief     This C++ code is used for image pixel format conversion.
\version   1.1.2604.9011
\date      2026.4.1
*/
//------------------------------------------------------------
#pragma once

#include <GxIAPI.h>
#include <DxImageProc.h>

//----------------------------------------------------------------------------------
/**
\brief  Determine whether the current pixel format is black and white.
\param  emPixelFormat[in]   pixel format
\return true: black pixelformat  false: color image
*/
//----------------------------------------------------------------------------------
bool IsGrayPixelFormat(GX_PIXEL_FORMAT_ENTRY emPixelFormat);

//----------------------------------------------------------------------------------
/**
\brief     Obtain the optimal Bit depth through GX_PIXEL_FORMAT_ENTRY
\param     emPixelFormatEntry image pixel format
\return    optimal bit
*/
//----------------------------------------------------------------------------------
DX_VALID_BIT GetBestValidBit(GX_PIXEL_FORMAT_ENTRY emPixelFormatEntry);

//----------------------------------------------------------------------------------
/**
\brief     Convert frame date to target pixel format
\param     pFrameBuffer     Image data structure pointer
\param     pDesImageBuf     Target buffer pointer
\param     nDesImageLen     Target buffer length
\param     emDesPixelFormat Target pixel format
\return    If an exception occurs during conversion, return an error; otherwise, return DX_OK.
*/
//----------------------------------------------------------------------------------
VxInt32 PixelFormatConvert(PGX_FRAME_DATA_EX pFrameBuffer, unsigned char *pDesImageBuf, uint32_t *pDesImageLen, GX_PIXEL_FORMAT_ENTRY emDesPixelFormat);