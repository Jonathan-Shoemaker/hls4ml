#ifndef NNET_CONV2DTRANSPOSE_STREAM_H
#define NNET_CONV2DTRANSPOSE_STREAM_H

#include "ap_shift_reg.h"
#include "nnet_conv_stream.h"
#include "nnet_common.h"
#include "hls_stream.h"

namespace nnet {

template <class data_T, typename CONFIG_T>
void kernel_shift_tr_2d(
    typename data_T::value_type shift_buffer[CONFIG_T::trfilt_height][CONFIG_T::n_chan],
    typename data_T::value_type kernel_window[CONFIG_T::trfilt_width * CONFIG_T::trfilt_height * CONFIG_T::n_chan]
) {
    #pragma HLS inline
        
    // Shift kernel_window by one step to the left (manual shift operation)
    static const int filt_width = CONFIG_T::trfilt_width - 1;
    KernelShiftWidth: for (int i_iw = 0; i_iw < filt_width; i_iw++) {
        #pragma HLS PIPELINE II = 1
        KernelShiftHeight: for (unsigned i_ih = 0; i_ih < CONFIG_T::trfilt_height; i_ih++) {
            KernelShiftChannel: for (unsigned i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
            // Shift every element in kernel_window to the left
                kernel_window[i_ih * CONFIG_T::trfilt_width * CONFIG_T::n_chan + i_iw * CONFIG_T::n_chan + i_ic] = kernel_window[i_ih * CONFIG_T::trfilt_width * CONFIG_T::n_chan + (i_iw + 1) * CONFIG_T::n_chan + i_ic];
            }
        }
    }

    // Insert shift_buffer column into right-most column of kernel
    static const int lastheight = (CONFIG_T::trfilt_width - 1) * CONFIG_T::n_chan;
    KernelPushHeight: for (int i_ih = 0; i_ih < CONFIG_T::trfilt_height; i_ih++) {
        #pragma HLS UNROLL
        KernelPushChannel: for (int i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
            kernel_window[lastheight + i_ih * CONFIG_T::trfilt_width * CONFIG_T::n_chan + i_ic] = shift_buffer[i_ih][i_ic];
        }
    }
}

template <class data_T, typename CONFIG_T>
void shift_line_buffer_tr(const data_T& in_elem, 
                    ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width> line_buffer[MAX(CONFIG_T::trfilt_height - 1,1)][CONFIG_T::n_chan],
                    typename data_T::value_type kernel_window[CONFIG_T::trfilt_height * CONFIG_T::trfilt_width * CONFIG_T::n_chan]
) {
    
    #pragma HLS PIPELINE

    // Temporary buffer for popped (shifted) elements
    typename data_T::value_type shift_buffer[CONFIG_T::trfilt_height][CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable = shift_buffer complete dim = 0

    UpdateBuffer: for (int i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
        #pragma HLS UNROLL

        // Insert pixel(s) at end of shift buffer
        shift_buffer[CONFIG_T::trfilt_height - 1][i_ic] = in_elem[i_ic];
    }

    LineBufferDataIn: for (int i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
        // Shift the shift buffer into the line buffer
        LineBufferShift: for (unsigned i_ih = 1; i_ih < CONFIG_T::trfilt_height; i_ih++) {
            #pragma HLS UNROLL
            typename data_T::value_type pop_elem = 
                line_buffer[i_ih - 1][i_ic].shift(shift_buffer[CONFIG_T::trfilt_height - i_ih][i_ic]); // Shift the line buffer, return the popped pixel
            shift_buffer[CONFIG_T::trfilt_height - i_ih - 1][i_ic] = pop_elem; // Popped element placed back into shift_buffer, one row up.
        }
    }
    kernel_shift_tr_2d<data_T, CONFIG_T>(shift_buffer, kernel_window);
}

template<typename CONFIG_T>
void load_tr_kern_weights(
    typename CONFIG_T::weight_t weights[CONFIG_T::trfilt_height * CONFIG_T::trfilt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t kernel_weights[
        CONFIG_T::n_filt * CONFIG_T::trfilt_height * CONFIG_T::trfilt_width * CONFIG_T::n_chan
    ],
    const int weight_x_start,
    const int weight_y_start
)
{
    int x_indices[CONFIG_T::trfilt_width];
    int y_indices[CONFIG_T::trfilt_height];
    for (int step = 0; step < CONFIG_T::trfilt_width; step++) {
        x_indices[step] = weight_x_start - step * CONFIG_T::stride_width;
    }
    for (int step = 0; step < CONFIG_T::trfilt_height; step++) {
        y_indices[step] = weight_y_start - step * CONFIG_T::stride_height;
    }

    WeightsLoop: for (int x_step = 0; x_step < CONFIG_T::trfilt_width; x_step++) {
        #pragma HLS UNROLL 
        #pragma HLS PIPELINE
        for (int y_step = 0; y_step < CONFIG_T::trfilt_height; y_step++) {
            #pragma HLS UNROLL 
            #pragma HLS PIPELINE
            for (int filt_ind = 0; filt_ind < CONFIG_T::n_filt; filt_ind++) {
                #pragma HLS UNROLL 
                #pragma HLS PIPELINE
                for (int chan_ind = 0; chan_ind < CONFIG_T::n_chan; chan_ind++) {
                    #pragma HLS UNROLL 
                    #pragma HLS PIPELINE
                    if (x_indices[x_step] >= CONFIG_T::filt_width || y_indices[y_step] >= CONFIG_T::filt_height) {
                        kernel_weights[
                            filt_ind * CONFIG_T::trfilt_height * CONFIG_T::trfilt_width * CONFIG_T::n_chan +
                            y_step * CONFIG_T::trfilt_width * CONFIG_T::n_chan + 
                            x_step * CONFIG_T::n_chan + chan_ind
                        ] = 0;
                    }
                    else {
                        kernel_weights[
                            filt_ind * CONFIG_T::trfilt_height * CONFIG_T::trfilt_width * CONFIG_T::n_chan +
                            y_step * CONFIG_T::trfilt_width * CONFIG_T::n_chan + 
                            x_step * CONFIG_T::n_chan + chan_ind
                        ] = weights[
                            y_indices[y_step] * CONFIG_T::filt_width * CONFIG_T::n_filt * CONFIG_T::n_chan +
                            x_indices[x_step] * CONFIG_T::n_filt * CONFIG_T::n_chan + 
                            filt_ind * CONFIG_T::n_chan + chan_ind
                        ];
                    }
                }
            }
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void compute_output_buffer_tr_2d(
    const data_T& in_elem,
    ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width> line_buffer[MAX(CONFIG_T::trfilt_height-1, 1)][CONFIG_T::n_chan],
    hls::stream<res_T> &res_stream,
    typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
)
{
    #pragma HLS INLINE

    //Counters
    static int pX = 0; //pixel counters
    static int pY = 0;

    static typename data_T::value_type kernel_data[CONFIG_T::trfilt_height * CONFIG_T::trfilt_width * CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=kernel_data complete

    typename CONFIG_T::weight_t kernel_weights[
        CONFIG_T::n_filt * CONFIG_T::trfilt_height * CONFIG_T::trfilt_width * CONFIG_T::n_chan
    ];

    typename res_T::value_type res_out[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=res_out complete dim = 0

    static typename res_T::value_type output_buffer[
        CONFIG_T::in_width*CONFIG_T::stride_width*CONFIG_T::stride_height*CONFIG_T::n_filt
    ];

    res_T res_pack;
    #pragma HLS DATA_PACK variable = res_pack

    //Add pixel to the buffer
    nnet::shift_line_buffer_tr<data_T, CONFIG_T>(in_elem, line_buffer, kernel_data);

    int weight_x_start = CONFIG_T::stride_width * (CONFIG_T::trfilt_width-1);
    int weight_y_start = CONFIG_T::stride_height * (CONFIG_T::trfilt_height-1);

    WidthStrideLoop: for (int h_idx = 0; h_idx < CONFIG_T::stride_height; h_idx++) {
        weight_x_start = CONFIG_T::stride_height * (CONFIG_T::trfilt_width-1);
        HeightStrideLoop: for (int w_idx = 0; w_idx < CONFIG_T::stride_width; w_idx++) {
            load_tr_kern_weights<CONFIG_T>(
                weights, kernel_weights, weight_x_start, weight_y_start
            );

            if (CONFIG_T::strategy == nnet::latency) {
                dense_latency<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::mult_config>(
                    kernel_data, res_out, kernel_weights, biases
                );
            } else {
                dense_resource<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::mult_config>(
                    kernel_data, res_out, kernel_weights, biases
                );
            }

            BufferOutputLoop: for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
                output_buffer[
                    (pX*CONFIG_T::stride_width+w_idx)*CONFIG_T::stride_height*CONFIG_T::n_filt + 
                    h_idx*CONFIG_T::n_filt + i_ic
                ] = res_out[i_ic];
            }

            weight_x_start++;
        }
        weight_y_start++;
    }

    //Counter Housekeeping
    if (pX + 1 == CONFIG_T::in_width) //HAVE TO THINK ABOUT oX, oY STUFF. NOT AS EASY AS INCREMENTING
    {
        pX = 0;
        //write all of the buffered output
        for (int h_idx = 0; h_idx < CONFIG_T::stride_height; h_idx++) {
            if (pY*CONFIG_T::stride_height + h_idx >= CONFIG_T::pad_top && 
                pY*CONFIG_T::stride_height +h_idx < CONFIG_T::pad_top + CONFIG_T::out_height) {
                for (int oX = CONFIG_T::pad_left; oX < CONFIG_T::pad_left + CONFIG_T::out_width; oX++) {
                    CastLoop: for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
                        #pragma HLS UNROLL
                        res_pack[i_ic] = output_buffer[
                            oX*CONFIG_T::stride_height*CONFIG_T::n_filt  + 
                            h_idx*CONFIG_T::n_filt + i_ic
                        ];
                    }
                    res_stream.write(res_pack);
                }
            }
        }

        if (pY + 1 == CONFIG_T::in_height) {
            pY = 0;
        } else {
            pY = pY + 1;
        }
    } else {
        pX = pX + 1;
    }

}

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_transpose_buffer_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{
    static ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width> line_buffer[MAX(CONFIG_T::trfilt_height-1, 1)][CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable = line_buffer complete dim = 2

    ReadInputHeight: for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            #pragma HLS LOOP_FLATTEN
            if (CONFIG_T::strategy == nnet::latency) {
                #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            }
            compute_output_buffer_tr_2d<data_T, res_T, CONFIG_T>(data.read(), line_buffer, res, weights, biases);
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_transpose_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t  biases[CONFIG_T::n_filt]
)
{
    #pragma HLS INLINE region
    switch(CONFIG_T::implementation) {
        case conv_implementation::linebuffer:
            conv_2d_transpose_buffer_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
            break;
    }
}

}
#endif
