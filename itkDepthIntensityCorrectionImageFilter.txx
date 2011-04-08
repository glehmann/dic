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
#ifndef __itkDepthIntensityCorrectionImageFilter_txx
#define __itkDepthIntensityCorrectionImageFilter_txx

#include "itkDepthIntensityCorrectionImageFilter.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkHistogram.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkProgressReporter.h"

namespace itk
{
template< class TInputImage, class TOutputImage >
void
DepthIntensityCorrectionImageFilter< TInputImage, TOutputImage >
::GenerateInputRequestedRegion()
{
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // We need all the input.
  InputImagePointer input = const_cast< InputImageType * >( this->GetInput() );
  if ( !input )
    {
    return;
    }
  input->SetRequestedRegion( input->GetLargestPossibleRegion() );
}

template< class TInputImage, class TOutputImage >
void
DepthIntensityCorrectionImageFilter< TInputImage, TOutputImage >
::EnlargeOutputRequestedRegion(DataObject *)
{
  this->GetOutput()
  ->SetRequestedRegion( this->GetOutput()->GetLargestPossibleRegion() );
}

template< class TInputImage, class TOutputImage >
int
DepthIntensityCorrectionImageFilter< TInputImage, TOutputImage >
::SplitRequestedRegion(int i, int num, RegionType & splitRegion)
{
  // Get the output pointer
  OutputImageType *outputPtr = this->GetOutput();

  const typename TOutputImage::SizeType & requestedRegionSize =
    outputPtr->GetRequestedRegion().GetSize();

  int splitAxis;
  typename TOutputImage::IndexType splitIndex;
  typename TOutputImage::SizeType splitSize;

  // Initialize the splitRegion to the output requested region
  splitRegion = outputPtr->GetRequestedRegion();
  splitIndex = splitRegion.GetIndex();
  splitSize = splitRegion.GetSize();

  // split on the outermost dimension available
  // and avoid the current dimension
  splitAxis = m_Dimension;
  if ( requestedRegionSize[splitAxis] == 1 )
    { // cannot split
    itkDebugMacro("  Cannot Split");
    return 1;
    }

  // determine the actual number of pieces that will be generated
  typename TOutputImage::SizeType::SizeValueType range = requestedRegionSize[splitAxis];
  int valuesPerThread = (int)vcl_ceil(range / (double)num);
  int maxThreadIdUsed = (int)vcl_ceil(range / (double)valuesPerThread) - 1;

  // Split the region
  if ( i < maxThreadIdUsed )
    {
    splitIndex[splitAxis] += i * valuesPerThread;
    splitSize[splitAxis] = valuesPerThread;
    }
  if ( i == maxThreadIdUsed )
    {
    splitIndex[splitAxis] += i * valuesPerThread;
    // last thread needs to process the "rest" dimension being split
    splitSize[splitAxis] = splitSize[splitAxis] - i * valuesPerThread;
    }

  // set the split region ivars
  splitRegion.SetIndex(splitIndex);
  splitRegion.SetSize(splitSize);

  itkDebugMacro("  Split Piece: " << splitRegion);

  return maxThreadIdUsed + 1;
}

template< class TInputImage, class TOutputImage >
void
DepthIntensityCorrectionImageFilter< TInputImage, TOutputImage >
::BeforeThreadedGenerateData()
{
  long nbOfThreads = this->GetNumberOfThreads();
  if ( itk::MultiThreader::GetGlobalMaximumNumberOfThreads() != 0 )
    {
    nbOfThreads = vnl_math_min( this->GetNumberOfThreads(), itk::MultiThreader::GetGlobalMaximumNumberOfThreads() );
    }
  // number of threads can be constrained by the region size, so call the
  // SplitRequestedRegion
  // to get the real number of threads which will be used
  typename TOutputImage::RegionType splitRegion;  // dummy region - just to call
                                                  // the following method
  nbOfThreads = this->SplitRequestedRegion(0, nbOfThreads, splitRegion);
  m_Barrier = Barrier::New();
  m_Barrier->Initialize(nbOfThreads);
  m_Quantiles.clear();
  m_Factors.clear();

  // get the min and max of the feature image, to use those value as the bounds
  // of our
  // histograms
  typedef MinimumMaximumImageCalculator< InputImageType > MinMaxCalculatorType;
  typename MinMaxCalculatorType::Pointer minMax = MinMaxCalculatorType::New();
  minMax->SetImage( this->GetInput() );
  minMax->Compute();

  m_Minimum = minMax->GetMinimum();
  m_Maximum = minMax->GetMaximum();
}

template< class TInputImage, class TOutputImage >
void
DepthIntensityCorrectionImageFilter< TInputImage, TOutputImage >
::ThreadedGenerateData(const RegionType & outputRegionForThread,
                       int threadId)
{
  typename TOutputImage::Pointer output = this->GetOutput();

  // set the progress reporter to deal with the number of lines
  ProgressReporter progress(this, threadId, outputRegionForThread.GetSize()[m_Dimension] * 2);

  typedef Statistics::Histogram< double > HistogramType;
  typename HistogramType::SizeType histogramSize;
  histogramSize.SetSize(1);
  histogramSize.Fill(256);

  typename HistogramType::MeasurementVectorType imageMin;
  imageMin.SetSize(1);
  imageMin.Fill(m_Minimum);

  typename HistogramType::MeasurementVectorType imageMax;
  imageMax.SetSize(1);
  imageMax.Fill(m_Maximum);

  typename HistogramType::MeasurementVectorType mv;
  mv.SetSize(1);

  IndexType idx = outputRegionForThread.GetIndex();
  SizeType size = outputRegionForThread.GetSize();
  size[m_Dimension] = 1;

  OffsetValueType bSlice = outputRegionForThread.GetIndex()[m_Dimension];
  OffsetValueType eSlice = outputRegionForThread.GetIndex()[m_Dimension] + outputRegionForThread.GetSize()[m_Dimension];
  for( OffsetValueType slice=bSlice; slice<eSlice; slice++ )
    {
    // first iterate over all the pixels of the slice to fill the histogram
    typename HistogramType::Pointer histogram = HistogramType::New();
    histogram->SetMeasurementVectorSize(1);
    histogram->SetClipBinsAtEnds(false);
    histogram->Initialize(histogramSize, imageMin, imageMax);
    idx[m_Dimension] = slice;
    RegionType sliceRegion(idx, size);
    typedef itk::ImageRegionConstIterator< InputImageType > IteratorType;
    IteratorType it( this->GetInput(), sliceRegion );
    while( !it.IsAtEnd() )
      {
      if( it.Get() >= m_Threshold )
        {
        mv[0] = it.Get();
        histogram->IncreaseFrequencyOfMeasurement(mv, 1);
        }
      ++it;
      }
    if( histogram->GetTotalFrequency() != 0 )
      {
      // compute the quantile
      double quantile = 0;
      double count = 0;
      for ( SizeValueType i = 0; i < histogram->Size(); i++ )
        {
        count += histogram->GetFrequency(i);

        if ( count >= ( histogram->GetTotalFrequency() * m_Rank ) )
          {
          quantile = histogram->GetMeasurementVector(i)[0];
          break;
          }
        }
      // store the value
      m_Quantiles[slice] = quantile;
      }
    else
      {
      // don't add the slice in the map
      }
    progress.CompletedPixel();
    }

  // wait for all the threads to complete their part
  this->Wait();
  
  // compute the factors to apply to each slice
  if( threadId == 0 )
    {
    double xb = 0;
    double yb = 0;
    double x2b = 0;
    double xyb = 0;
    for( typename MapType::const_iterator mit = m_Quantiles.begin(); mit!=m_Quantiles.end(); mit++ )
      {
      double x = mit->first;
      double y = vcl_log( mit->second );
//       std::cout << "quantile: " << mit->second << std::endl;
      xb += x;
      yb += y;
      x2b += x*x;
      xyb += x*y;
      }
    xb /= m_Quantiles.size();
    yb /= m_Quantiles.size();
    x2b /= m_Quantiles.size();
    xyb /= m_Quantiles.size();
    double Sxy = xyb - (xb*yb);
    double Sx2 = x2b - (xb*xb);

    double a = Sxy / Sx2;
    double b = yb - a * xb;
    
    double greatest = NumericTraits<double>::NonpositiveMin();
    for( typename MapType::const_iterator mit = m_Quantiles.begin(); mit!=m_Quantiles.end(); mit++ )
      {
      double x = mit->first;
      double y = vcl_exp( a * x + b );
      m_Factors[x] = y;
      greatest = std::max( greatest, y );
//       std::cout << "y: " << y << std::endl;
      }
    for( typename MapType::iterator mit = m_Factors.begin(); mit!=m_Factors.end(); mit++ )
      {
      mit->second = greatest / mit->second;
//       std::cout << "factor: " << mit->second << std::endl;
      }
    
    // we'll need the output data very soon
    this->AllocateOutputs();
    }
  // wait for the other threads to complete that part
  this->Wait();

  // generate the output with the rescaled output value
  for( OffsetValueType slice=bSlice; slice<eSlice; slice++ )
    {
    typedef itk::ImageRegionConstIterator< InputImageType > InIteratorType;
    typedef itk::ImageRegionIterator< OutputImageType > OutIteratorType;
    idx[m_Dimension] = slice;
    RegionType sliceRegion(idx, size);
    InIteratorType iit( this->GetInput(), sliceRegion );
    OutIteratorType oit( this->GetOutput(), sliceRegion );
    double factor = m_Factors[slice];
    while( !oit.IsAtEnd() )
      {
      double out = std::min( iit.Get() * factor, (double)NumericTraits<OutputPixelType>::max() );
      oit.Set( (OutputPixelType)out );
      ++iit;
      ++oit;
      }
    progress.CompletedPixel();
    }

}

template< class TInputImage, class TOutputImage >
void
DepthIntensityCorrectionImageFilter< TInputImage, TOutputImage >
::AfterThreadedGenerateData()
{
  m_Quantiles.clear();
  m_Factors.clear();
  m_Barrier = NULL;
}

template< class TInputImage, class TOutputImage >
void
DepthIntensityCorrectionImageFilter< TInputImage, TOutputImage >
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Threshold: " << static_cast< typename NumericTraits< InputPixelType >::PrintType >( m_Threshold ) << std::endl;
  os << indent << "Dimension: " << m_Dimension << std::endl;
  os << indent << "Rank: " << m_Rank << std::endl;
}
} // end namespace itk

#endif
