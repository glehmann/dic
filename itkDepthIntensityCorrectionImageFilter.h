/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __itkDepthIntensityCorrectionImageFilter_h
#define __itkDepthIntensityCorrectionImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include <map>
#include "itkProgressReporter.h"
#include "itkBarrier.h"

namespace itk
{
/**
 * \class DepthIntensityCorrectionImageFilter
 * \brief 
 *
 *
 * \sa ImageToImageFilter
 *
 */

template< class TInputImage, class TOutputImage=TInputImage >
class ITK_EXPORT DepthIntensityCorrectionImageFilter:
  public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /**
   * Standard "Self" & Superclass typedef.
   */
  typedef DepthIntensityCorrectionImageFilter                   Self;
  typedef ImageToImageFilter< TInputImage, TOutputImage > Superclass;

  /**
   * Types from the Superclass
   */
  typedef typename Superclass::InputImagePointer InputImagePointer;

  /**
   * Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same.
   */
  typedef typename TOutputImage::PixelType         OutputPixelType;
  typedef typename TOutputImage::InternalPixelType OutputInternalPixelType;
  typedef typename TInputImage::PixelType          InputPixelType;
  typedef typename TInputImage::InternalPixelType  InputInternalPixelType;
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TOutputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      TOutputImage::ImageDimension);
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      TInputImage::ImageDimension);

  /**
   * Image typedef support
   */
  typedef TInputImage                      InputImageType;
  typedef typename TInputImage::IndexType  IndexType;
  typedef typename TInputImage::SizeType   SizeType;
  typedef typename TInputImage::OffsetType OffsetType;

  typedef TOutputImage                      OutputImageType;
  typedef typename TOutputImage::RegionType RegionType;
  typedef typename TOutputImage::IndexType  OutputIndexType;
  typedef typename TOutputImage::SizeType   OutputSizeType;
  typedef typename TOutputImage::OffsetType OutputOffsetType;
  typedef typename TOutputImage::PixelType  OutputImagePixelType;

  /**
   * Smart pointer typedef support
   */
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /**
   * Run-time type information (and related methods)
   */
  itkTypeMacro(DepthIntensityCorrectionImageFilter, ImageToImageFilter);

  /**
   * Method for creation through the object factory.
   */
  itkNewMacro(Self);

  /**
   */
  itkSetMacro(Dimension, SizeValueType);
  itkGetConstMacro(Dimension, SizeValueType);

  /**
   */
  itkSetMacro(Rank, double);
  itkGetConstMacro(Rank, double);

  /**
   */
  itkSetMacro(Threshold, InputPixelType);
  itkGetConstMacro(Threshold, InputPixelType);

  
  enum {DIRECT=0, REGRESSION=1} Method;
  enum {QUANTILE=0, MEAN=1} Measure;
  
  /**
   */
  itkSetMacro(Method, int);
  itkGetConstMacro(Method, int);

  /**
   */
  itkSetMacro(Measure, int);
  itkGetConstMacro(Measure, int);

protected:
  DepthIntensityCorrectionImageFilter()
  {
    m_Dimension = ImageDimension - 1;
    m_Rank = 0.5;
    m_Threshold = NumericTraits<InputPixelType>::NonpositiveMin();
    m_Method = REGRESSION;
    m_Measure = QUANTILE;
  }

  virtual ~DepthIntensityCorrectionImageFilter() {}
  DepthIntensityCorrectionImageFilter(const Self &) {}
  void PrintSelf(std::ostream & os, Indent indent) const;

  /**
   * Standard pipeline methods.
   */
  void BeforeThreadedGenerateData();

  void AfterThreadedGenerateData();

  void ThreadedGenerateData(const RegionType & outputRegionForThread, int threadId);

  /** DepthIntensityCorrectionImageFilter needs the entire input. Therefore
   * it must provide an implementation GenerateInputRequestedRegion().
   * \sa ProcessObject::GenerateInputRequestedRegion(). */
  void GenerateInputRequestedRegion();

  /** DepthIntensityCorrectionImageFilter needs the entire slice for each thread */
  int SplitRequestedRegion(int i, int num, RegionType & splitRegion);

  /** DepthIntensityCorrectionImageFilter shouldn't be forced to produce the full output,
   * but it makes the management of the threads easier
   */
  void EnlargeOutputRequestedRegion( DataObject * itkNotUsed(output) );

private:
  SizeValueType                           m_Dimension;
  InputPixelType                          m_Threshold;
  double                                  m_Rank;
  int                                     m_Method;
  int                                     m_Measure;

  // used internaly
  typedef std::map< OffsetValueType, double > MapType;

  InputPixelType                          m_Minimum;
  InputPixelType                          m_Maximum;
  MapType                                 m_Measures;
  MapType                                 m_Factors;
  typename Barrier::Pointer               m_Barrier;
  int                                     m_NumberOfThreads;

  void Wait()
  {
    if ( m_NumberOfThreads > 1 )
      {
      m_Barrier->Wait();
      }
  }
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#if !defined( CABLE_CONFIGURATION )
#include "itkDepthIntensityCorrectionImageFilter.txx"
#endif
#endif

#endif
