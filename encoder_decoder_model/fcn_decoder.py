# -*- coding: utf-8 -*-
# @Time    : 18-12-07
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : fcn_decoder.py
# @IDE     : PyCharm Community Edition

import tensorflow as tf
from encoder_decoder_model import cnn_base_model
from encoder_decoder_model import vgg_encoder
from encoder_decoder_model import dense_encoder
"""
packing FCN 
"""

class FCNDecoder(cnn_base_model.CNNBaseModel):
    """
    packing FCN
    """
    def __init__(self, phase):
        super(FCNDecoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """
        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def decode(self, input_tensor_dict, decode_layer_list, name):
        """
        using deconv to decode the network and get pixel feature info

        :param input_tensor_dict:
        :param decode_layer_list: those layers which need to be decoded
                                  need to be written from deep to shallow
                                  eg. ['pool5', 'pool4', 'pool3']
        :param name:
        :return:
        """
        ret = dict()

        with tf.variable_scope(name):
            # score stage 1
            input_tensor = input_tensor_dict[decode_layer_list[0]]['data']

            score = self.conv2d(input_data=input_tensor, out_channel=64,
                                kernel_size=1, use_bias=False, name='score_origin')
            decode_layer_list = decode_layer_list[1:]
            for i in range(len(decode_layer_list)):
                deconv = self.deconv2d(input_data=score, out_channel=64, kernel_size=4,
                                       stride=2, use_bias=False, name='deconv_{:d}'.format(i + 1))
                input_tensor = input_tensor_dict[decode_layer_list[i]]['data']
                score = self.conv2d(input_data=input_tensor, out_channel=64,
                                    kernel_size=1, use_bias=False, name='score_{:d}'.format(i + 1))
                fused = tf.add(deconv, score, name='fuse_{:d}'.format(i + 1))
                score = fused

            deconv_final = self.deconv2d(input_data=score, out_channel=64, kernel_size=16,
                                         stride=8, use_bias=False, name='deconv_final')

            score_final = self.conv2d(input_data=deconv_final, out_channel=2,
                                      kernel_size=1, use_bias=False, name='score_final')

            ret['logits'] = score_final
            ret['deconv'] = deconv_final

        return ret


if __name__ == '__main__':

    vgg_encoder = vgg_encoder.VGG16Encoder(phase=tf.constant('train', tf.string))
    dense_encoder = dense_encoder.DenseEncoder(L=40, growth_rate=12,
                                               with_bc=True, phase='train', N=5)
    decoder = FCNDecoder(phase='train')

    in_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 256, 512, 3],
                               name='input')

    vgg_encode_ret = vgg_encoder.encode(in_tensor, name='vgg_encoder')
    dense_encode_ret = dense_encoder.encode(in_tensor, name='dense_encoder')
    decode_ret = decoder.decode(vgg_encode_ret, name='decoder',
                                decode_layer_list=['pool5',
                                                   'pool4',
                                                   'pool3'])
    print(decode_ret)