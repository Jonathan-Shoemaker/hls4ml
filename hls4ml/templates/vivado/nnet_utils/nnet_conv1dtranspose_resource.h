#ifndef NNET_CONV1DTRANSPOSE_RESOURCE_H_
#define NNET_CONV1DTRANSPOSE_RESOURCE_H_

#include "nnet_common.h"
#include "nnet_dense.h"

/*
Implementation comments:
  - have to update weights array as well as data array for each output
  - could probably be more clever so that weights update less (unsure of how important this is)
  - have to zero out the excess every time I think
*/

namespace nnet{

template<typename CONFIG_T>
void weights_trim(
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t row_weights[
        CONFIG_T::n_filt * CONFIG_T::trfilt_width * CONFIG_T::n_chan
    ],
    const int weight_start,
    const int cur_filt_width
)
{
    int row_indices[CONFIG_T::trfilt_width];
    for (int step = 0; step < CONFIG_T::trfilt_width; step++) {
        row_indices[step] = weight_start - step * CONFIG_T::stride_width;
    }

    KernelLoop:
    for (int step = 0; step < CONFIG_T::trfilt_width; step++) {
        #pragma HLS UNROLL 
        #pragma HLS PIPELINE
        
        for (int filt_ind = 0; filt_ind < CONFIG_T::n_filt; filt_ind++) {
            #pragma HLS UNROLL
            #pragma HLS PIPELINE
            for (int chan_ind = 0; chan_ind < CONFIG_T::n_chan; chan_ind++) {
                #pragma HLS UNROLL
                #pragma HLS PIPELINE
                if (step >= cur_filt_width) {
                    row_weights[filt_ind * CONFIG_T::trfilt_width * CONFIG_T::n_chan + 
                        step * CONFIG_T::n_chan + chan_ind] = 0;
                } else {
                    row_weights[filt_ind * CONFIG_T::trfilt_width * CONFIG_T::n_chan + 
                        step * CONFIG_T::n_chan + chan_ind] = 
                        weights[row_indices[step] * CONFIG_T::n_filt * CONFIG_T::n_chan + 
                            filt_ind * CONFIG_T::n_chan + chan_ind];
                }
            }
        }
    }
}

template<class data_T, typename CONFIG_T>
void im2col_start_width(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    data_T data_col[CONFIG_T::trfilt_width * CONFIG_T::n_chan],
    const int start_index,
    const int cur_filt_width
)
{
    int index = 0;
    KernelLoop:
    for (int kernel_col = 0; kernel_col < CONFIG_T::trfilt_width; kernel_col++) {
        #pragma HLS UNROLL
        #pragma HLS PIPELINE
        ChannelLoop:
        for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
            #pragma HLS UNROLL
            #pragma HLS PIPELINE
            int index_data = (start_index + kernel_col) * CONFIG_T::n_chan + channel;
            if (kernel_col >= cur_filt_width || index_data < 0 || 
                index_data >= CONFIG_T::in_width * CONFIG_T::n_chan) {
                data_col[index] = 0;
            } else {
                data_col[index] =  data[index_data];
            }
            index++;
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_1d_transpose_resource_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T res[CONFIG_T::out_width  * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_filt * CONFIG_T::filt_width * CONFIG_T::n_chan],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{

    const int nin = CONFIG_T::trfilt_width * CONFIG_T::n_chan;
    const int nout = CONFIG_T::n_filt;
    const int rufactor = CONFIG_T::reuse_factor;
    const int block_factor = DIV_ROUNDUP(nin*nout, rufactor);

    data_T data_col[CONFIG_T::trfilt_width * CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];
    int start_indices[CONFIG_T::out_width];
    int weight_starts[CONFIG_T::out_width];
    int filt_widths[CONFIG_T::out_width];

    typename CONFIG_T::weight_t row_weights[
        CONFIG_T::n_filt * CONFIG_T::trfilt_width * CONFIG_T::n_chan
    ];

    #pragma HLS ARRAY_PARTITION variable=data_col complete
    #pragma HLS ARRAY_PARTITION variable=res_col complete
    #pragma HLS ARRAY_PARTITION variable=start_indices complete 
    #pragma HLS ARRAY_PARTITION variable=weight_starts complete
    #pragma HLS ARRAY_PARTITION variable=filt_widths complete
    // #pragma HLS RESOURCE variable=row_weights core=RAM_2P_BRAM

    // calculate the start indices first
    CalcIndexLoop:
    for (int out_ind = 0; out_ind < CONFIG_T::out_width; out_ind++) {
        #pragma HLS PIPELINE
        start_indices[out_ind] = 
            (out_ind + CONFIG_T::pad_left - CONFIG_T::filt_width)/CONFIG_T::stride_width + 1;
        weight_starts[out_ind] = 
            out_ind + CONFIG_T::pad_left - CONFIG_T::stride_width*start_indices[out_ind];
        filt_widths[out_ind] = MIN(
            weight_starts[out_ind] / CONFIG_T::stride_width + 1, 
            CONFIG_T::in_width - start_indices[out_ind]
        );
    }

    #pragma HLS ARRAY_RESHAPE variable=row_weights block factor=block_factor
    
    ColLoop:
    for (int out_ind = 0; out_ind < CONFIG_T::out_width; out_ind++) {
        #pragma HLS PIPELINE
        weights_trim<CONFIG_T>(
            weights, row_weights, weight_starts[out_ind], filt_widths[out_ind]
        );
        im2col_start_width<data_T, CONFIG_T>(
            data, data_col, start_indices[out_ind], filt_widths[out_ind]
        );
        dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(
            data_col, res_col, row_weights, biases
        );

        for (int j = 0; j < CONFIG_T::n_filt; j++) {
            res[out_ind * CONFIG_T::n_filt + j] = res_col[j];
        }
    }

}

template<class data_T, class res_T, typename CONFIG_T>
void conv_1d_transpose_resource_by_kernel_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T res[CONFIG_T::out_width  * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_filt * CONFIG_T::filt_width * CONFIG_T::n_chan],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{
    data_T data_col[CONFIG_T::trfilt_width * CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];

    #pragma HLS ARRAY_PARTITION variable=data_col complete
    #pragma HLS ARRAY_PARTITION variable=res_col complete

    typename CONFIG_T::weight_t row_weights[
        CONFIG_T::n_filt * CONFIG_T::trfilt_width * CONFIG_T::n_chan
    ];

    //loop over the output cells, compute each one separately
    int start_index = 0;

    //kern_ind is a weird variable but should get the job done
    ColLoop:
    for (int kern_ind = 0; kern_ind < CONFIG_T::stride_width; kern_ind++) {
        #pragma HLS UNROLL
        #pragma HLS PIPELINE
        int num_str = (CONFIG_T::filt_width - kern_ind - 1)/CONFIG_T::stride_width;
        int weight_start = num_str * CONFIG_T::stride_width + kern_ind;
        int cur_filt_width = weight_start / CONFIG_T::stride_width+1;
        weights_trim<CONFIG_T>(weights, row_weights, weight_start, cur_filt_width);

        start_index = 1 - cur_filt_width;
        start_index += DIV_ROUNDUP(CONFIG_T::pad_left - kern_ind, CONFIG_T::stride_width);
        
        int out_ind_start = kern_ind + 
            DIV_ROUNDUP(CONFIG_T::pad_left - kern_ind, CONFIG_T::stride_width)*CONFIG_T::stride_width;

        KernelLoop:
        for (int out_ind = out_ind_start; out_ind < CONFIG_T::out_width + CONFIG_T::pad_left; 
            out_ind+=CONFIG_T::stride_width) {
            #pragma HLS PIPELINE

            im2col_start_width<data_T, CONFIG_T>(data, data_col, start_index, cur_filt_width);
            dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(
                data_col, res_col, row_weights, biases
            );
            for (int j = 0; j < CONFIG_T::n_filt; j++) {
                res[(out_ind - CONFIG_T::pad_left) * CONFIG_T::n_filt + j] = res_col[j];
            }
            start_index++;
        }
    }
}

}
#endif
