#ifndef NNET_CONV1DTRANSPOSE_RESOURCE_H_
#define NNET_CONV1DTRANSPOSE_RESOURCE_H_

#include "nnet_common.h"
#include "nnet_dense.h"

/*
Implementation comments:
  - have to update weights array as well as data array for each output
  - could probably be more clever so that weights update less often (unsure of how important this is)
  - both of these arrays are bigger than they need to be, because their size changes
    - could probably trim the size more cleverly
  - have to zero out the excess every time I think
*/

namespace nnet{

template<typename CONFIG_T>
void weights_trim(
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t row_weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    const int weight_start,
    const int cur_filt_width
)
{
    int row_index = weight_start;
    KernelLoop:
    for (int step = 0; step < cur_filt_width; step++) {
        for (int inner_ind = 0; inner_ind < CONFIG_T::n_chan * CONFIG_T::n_filt; inner_ind++) {
            row_weights[step * CONFIG_T::n_chan * CONFIG_T::n_filt + inner_ind] = 
                weights[row_index * CONFIG_T::n_chan * CONFIG_T::n_filt + inner_ind];
        }
        row_index -= CONFIG_T::stride_width;
    }
    ZeroLoop:
    for (int step = 0; step < CONFIG_T::filt_width-cur_filt_width; step++) {
        for (int inner_ind = 0; inner_ind < CONFIG_T::n_chan * CONFIG_T::n_filt; inner_ind++) {
            row_weights[step * CONFIG_T::n_chan * CONFIG_T::n_filt + inner_ind] = 0;
        }
    }
}

template<class data_T, typename CONFIG_T>
void im2col_start_width(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    data_T data_col[CONFIG_T::filt_width * CONFIG_T::n_chan],
    const int start_index,
    const int cur_filt_width
)
{
    int index = 0;
    KernelLoop:
    for (int kernel_col = 0; kernel_col < cur_filt_width; kernel_col++) {

        ChannelLoop:
        for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
            data_col[index] =  (start_index + kernel_col) * CONFIG_T::n_chan + channel;
        }
        index++;
    }
    for (int kernel_col = cur_filt_width; kernel_col < CONFIG_T::filt_width; kernel_col++) {   
        for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
            data_col[index] = 0;
        }
        index++;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_1d_transpose_resource_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T res[CONFIG_T::out_width  * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{
    
    //for now, maybe just implement straight-up matrix multiplication?
    //    -> makes sure the weights and kernel are set up as we expect, then we can proceed with debugging
    //    -> implements some of our indexing logic (should help debug)
    //    -> indexing may be more difficult than I first thought
    int start_index = 0;
    for (int i = CONFIG_T::pad_left; i < CONFIG_T::out_width+CONFIG_T::pad_left; i++) {
        if (i > start_index * CONFIG_T::stride_width + CONFIG_T::filt_width-1) {
            start_index++;
        }
        int weight_start = i - CONFIG_T::stride_width*start_index;
        int cur_filt_width = weight_start / CONFIG_T::stride_width + 1;
        cur_filt_width = MIN(cur_filt_width, CONFIG_T::in_width-start_index);
        for (int j = 0; j < CONFIG_T::n_filt; j++) {
            //compute this output cell
            int cur_ind = (i-CONFIG_T::pad_left) * CONFIG_T::n_filt + j;
            res[cur_ind] = 0;
            for (int k = 0; k < CONFIG_T::n_chan; k++) {
                for (int ki = 0; ki < cur_filt_width; ki++) {
                    int filt_ind = weight_start - ki * CONFIG_T::stride_width;
                    res[cur_ind] += data[(start_index + ki) * CONFIG_T::n_chan + k] * 
                        weights[filt_ind*CONFIG_T::n_chan*CONFIG_T::n_filt + k * CONFIG_T::n_filt + j];
                }
            }
        }
    }

    return;

    data_T data_col[CONFIG_T::filt_width * CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];
    typename CONFIG_T::weight_t row_weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt];
    //loop over the output cells, compute each one separately
    
    start_index = 0;

    ColLoop:
    for (int out_ind = CONFIG_T::pad_left; out_ind < CONFIG_T::out_width + CONFIG_T::pad_left; out_ind++) {
        //corresponds to res_col index out_ind - pad_left
        if (out_ind > start_index*CONFIG_T::stride_width + CONFIG_T::filt_width - 1) {
           start_index++;
        }
        int weight_start = out_ind - CONFIG_T::stride_width*start_index + 1;
        int cur_filt_width = weight_start / CONFIG_T::stride_width + 1;
        cur_filt_width = MIN(cur_filt_width, CONFIG_T::n_chan-weight_start);

        weights_trim<CONFIG_T>(weights, row_weights, weight_start, cur_filt_width);
        im2col_start_width<data_T, CONFIG_T>(data, data_col, start_index, cur_filt_width);

        dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, row_weights, biases);

        for (int j = 0; j < CONFIG_T::n_filt; j++) {
            res[(out_ind-CONFIG_T::pad_left) * CONFIG_T::n_filt + j] = res_col[j];
        }
    }
}

}
#endif
