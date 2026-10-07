//-------------------------------------------------------------
/**
\file      Common.cpp
\brief     This C++ code is used for image pixel format conversion.
\version   1.1.2604.9011
\date      2026.4.01
*/
//------------------------------------------------------------
#include "Common.h"

#define DX_IMAGE_PROC_BREAK(emStatus) \
    if (DX_OK != emStatus) 				\
    {                                  \
        break;               \
    }

//----------------------------------------------------------------------------------
/**
\brief  Determine whether the current pixel format is black and white.
\param  emPixelFormat[in]   pixel format
\return true: black pixelformat  false: color image
*/
//----------------------------------------------------------------------------------
bool IsGrayPixelFormat(GX_PIXEL_FORMAT_ENTRY emPixelFormat)
{
    bool bGrayPixelFormat = false;
    switch (emPixelFormat)
    {
    case GX_PIXEL_FORMAT_R8:
    case GX_PIXEL_FORMAT_G8:
    case GX_PIXEL_FORMAT_B8:
    case GX_PIXEL_FORMAT_MONO8:
    case GX_PIXEL_FORMAT_MONO8_SIGNED:
    case GX_PIXEL_FORMAT_MONO10:
	case GX_PIXEL_FORMAT_MONO10_P:
    case GX_PIXEL_FORMAT_MONO10_PACKED:
    case GX_PIXEL_FORMAT_MONO12:
	case GX_PIXEL_FORMAT_MONO12_P:
    case GX_PIXEL_FORMAT_MONO12_PACKED:
    case GX_PIXEL_FORMAT_MONO14:
	case GX_PIXEL_FORMAT_MONO14_P:
    case GX_PIXEL_FORMAT_MONO16:
        bGrayPixelFormat = true;
        break;
    default:
        break;
    }

    return bGrayPixelFormat;
}

//----------------------------------------------------------------------------------
/**
\brief     Obtain the optimal Bit depth through GX_PIXEL_FORMAT_ENTRY
\param     emPixelFormatEntry image pixel format
\return    optimal bit
*/
//----------------------------------------------------------------------------------
DX_VALID_BIT GetBestValidBit(GX_PIXEL_FORMAT_ENTRY emPixelFormatEntry)
{
	DX_VALID_BIT emValidBits = DX_BIT_0_7;
	switch (emPixelFormatEntry)
	{
	case GX_PIXEL_FORMAT_MONO8:
	case GX_PIXEL_FORMAT_BAYER_GR8:
	case GX_PIXEL_FORMAT_BAYER_RG8:
	case GX_PIXEL_FORMAT_BAYER_GB8:
	case GX_PIXEL_FORMAT_BAYER_BG8:
	case GX_PIXEL_FORMAT_BGR8:
	case GX_PIXEL_FORMAT_RGB8:
	case GX_PIXEL_FORMAT_RGBA8:
	case GX_PIXEL_FORMAT_BGRA8:
	case GX_PIXEL_FORMAT_ARGB8:
	case GX_PIXEL_FORMAT_ABGR8:
	case GX_PIXEL_FORMAT_R8:
	case GX_PIXEL_FORMAT_G8:
	case GX_PIXEL_FORMAT_B8:
	case GX_PIXEL_FORMAT_YUV444_8:
	case GX_PIXEL_FORMAT_YUV422_8:
	case GX_PIXEL_FORMAT_YUV411_8:
	case GX_PIXEL_FORMAT_YCBCR444_8:
	case GX_PIXEL_FORMAT_YCBCR422_8:
	case GX_PIXEL_FORMAT_YCBCR411_8:
	case GX_PIXEL_FORMAT_YCBCR601_444_8:
	case GX_PIXEL_FORMAT_YCBCR601_422_8:
	case GX_PIXEL_FORMAT_YCBCR601_411_8:

	{
		emValidBits = DX_BIT_0_7;
		break;
	}
	case GX_PIXEL_FORMAT_MONO10:
	case GX_PIXEL_FORMAT_MONO10_P:
	case GX_PIXEL_FORMAT_MONO10_PACKED:
	case GX_PIXEL_FORMAT_BAYER_GR10:
	case GX_PIXEL_FORMAT_BAYER_RG10:
	case GX_PIXEL_FORMAT_BAYER_GB10:
	case GX_PIXEL_FORMAT_BAYER_BG10:
	case GX_PIXEL_FORMAT_BAYER_BG10_PACKED:
	case GX_PIXEL_FORMAT_BAYER_GB10_PACKED:
	case GX_PIXEL_FORMAT_BAYER_GR10_PACKED:
	case GX_PIXEL_FORMAT_BAYER_RG10_PACKED:
	case GX_PIXEL_FORMAT_BAYER_BG10_P:
	case GX_PIXEL_FORMAT_BAYER_GB10_P:
	case GX_PIXEL_FORMAT_BAYER_GR10_P:
	case GX_PIXEL_FORMAT_BAYER_RG10_P:
	case GX_PIXEL_FORMAT_RGB10:
	case GX_PIXEL_FORMAT_BGR10:
	{
		emValidBits = DX_BIT_2_9;
		break;
	}
	case GX_PIXEL_FORMAT_MONO12:
	case GX_PIXEL_FORMAT_MONO12_P:
	case GX_PIXEL_FORMAT_MONO12_PACKED:
	case GX_PIXEL_FORMAT_BAYER_GR12:
	case GX_PIXEL_FORMAT_BAYER_RG12:
	case GX_PIXEL_FORMAT_BAYER_GB12:
	case GX_PIXEL_FORMAT_BAYER_BG12:
	case GX_PIXEL_FORMAT_BAYER_BG12_PACKED:
	case GX_PIXEL_FORMAT_BAYER_GB12_PACKED:
	case GX_PIXEL_FORMAT_BAYER_GR12_PACKED:
	case GX_PIXEL_FORMAT_BAYER_RG12_PACKED:
	case GX_PIXEL_FORMAT_BAYER_BG12_P:
	case GX_PIXEL_FORMAT_BAYER_GB12_P:
	case GX_PIXEL_FORMAT_BAYER_GR12_P:
	case GX_PIXEL_FORMAT_BAYER_RG12_P:
	case GX_PIXEL_FORMAT_RGB12:
	case GX_PIXEL_FORMAT_BGR12:
	case GX_PIXEL_FORMAT_RGBA12:
	case GX_PIXEL_FORMAT_RGBA12_P:
	case GX_PIXEL_FORMAT_BGRA12:
	case GX_PIXEL_FORMAT_BGRA12_P:
	{
		emValidBits = DX_BIT_4_11;
		break;
	}
	case GX_PIXEL_FORMAT_MONO14:
	case GX_PIXEL_FORMAT_MONO14_P:
	case GX_PIXEL_FORMAT_BAYER_GR14:
	case GX_PIXEL_FORMAT_BAYER_RG14:
	case GX_PIXEL_FORMAT_BAYER_GB14:
	case GX_PIXEL_FORMAT_BAYER_BG14:
	case GX_PIXEL_FORMAT_BAYER_GR14_P:
	case GX_PIXEL_FORMAT_BAYER_RG14_P:
	case GX_PIXEL_FORMAT_BAYER_GB14_P:
	case GX_PIXEL_FORMAT_BAYER_BG14_P:
	case GX_PIXEL_FORMAT_RGB14:
	case GX_PIXEL_FORMAT_BGR14:
	{
		emValidBits = DX_BIT_6_13;
		break;
	}
	case GX_PIXEL_FORMAT_MONO16:
	case GX_PIXEL_FORMAT_BAYER_GR16:
	case GX_PIXEL_FORMAT_BAYER_RG16:
	case GX_PIXEL_FORMAT_BAYER_GB16:
	case GX_PIXEL_FORMAT_BAYER_BG16:
	case GX_PIXEL_FORMAT_RGB16:
	case GX_PIXEL_FORMAT_BGR16:
	{
		emValidBits = DX_BIT_8_15;
		break;
	}
	default:
	{	
		break;
	}
	}
	return emValidBits;
}

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
VxInt32 PixelFormatConvert(PGX_FRAME_DATA_EX pFrameBuffer, unsigned char *pDesImageBuf, uint32_t *pDesImageLen, GX_PIXEL_FORMAT_ENTRY emDesPixelFormat)
{
	DX_IMAGE_FORMAT_CONVERT_HANDLE hConvertHandle = NULL;
	VxInt32 emStatus = DX_OK;
		
	do
	{
		if((NULL == pFrameBuffer) || (NULL == pDesImageBuf) || ( NULL == pDesImageLen))
		{
			return DX_PARAMETER_INVALID;
		}
		
		emStatus = DxImageFormatConvertCreate( &hConvertHandle);
		DX_IMAGE_PROC_BREAK(emStatus);
		
		int32_t nDesBufferSize = 0;
		emStatus = DxImageFormatConvertGetBufferSizeForConversion(hConvertHandle, 
			emDesPixelFormat, 
			pFrameBuffer->nWidth, 
			pFrameBuffer->nHeight, 
			&nDesBufferSize);
		DX_IMAGE_PROC_BREAK(emStatus);
	
		if( *pDesImageLen < nDesBufferSize)
		{
			emStatus = DX_PARAMETER_INVALID;
			break;
		}
	
		if( emDesPixelFormat == pFrameBuffer->nPixelFormat)
		{
			memcpy( pDesImageBuf, (void*)pFrameBuffer->pImgBuf, nDesBufferSize);
			*pDesImageLen = nDesBufferSize;
	
			break;
		}
	
		emStatus = DxImageFormatConvertSetOutputPixelFormat( hConvertHandle, emDesPixelFormat);
		DX_IMAGE_PROC_BREAK(emStatus);
		
		DX_VALID_BIT emValidBit = GetBestValidBit( (GX_PIXEL_FORMAT_ENTRY)pFrameBuffer->nPixelFormat);
		emStatus = DxImageFormatConvertSetValidBits( hConvertHandle, emValidBit);
		DX_IMAGE_PROC_BREAK(emStatus);
	
		emStatus = DxImageFormatConvert( hConvertHandle, 
			(void*)pFrameBuffer->pImgBuf, 
			pFrameBuffer->nImgSize, 
			pDesImageBuf, 
			*pDesImageLen, 
			(GX_PIXEL_FORMAT_ENTRY)pFrameBuffer->nPixelFormat, 
			pFrameBuffer->nWidth, 
			pFrameBuffer->nHeight, 
			false);
		DX_IMAGE_PROC_BREAK(emStatus);
	
		*pDesImageLen = nDesBufferSize;
	}while (false);
	
	if( NULL != hConvertHandle)
	{
		emStatus = DxImageFormatConvertDestroy( hConvertHandle);
		hConvertHandle = NULL;
	}
	
	return emStatus;
}
