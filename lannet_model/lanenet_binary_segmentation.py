# -*- coding: utf-8 -*-
# @Time    : 18-12-10
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : lanenet_binary_segmentation.py
# @IDE     : PyCharm Community Edition
"""
implement image binary segmentation
"""

import tensorflow as tf

from encoder_decoder_model import vgg_encoder
from encoder_decoder_model import fcn_decoder
from encoder_decoder_model import dense_encoder
from encoder_decoder_model import cnn_base_model


class LaneNetBinarySeg(cnn_base_model.CNNBaseModel):
    """
    implement image binary segmentation
    """
    def __init__(self, phase, net_flag='vgg'):
        """

        :param phase:
        :param net_flag:
        """
        super(LaneNetBinarySeg, self).__init__()
        self._phase = phase
        self._net_flag = net_flag
        if net_flag == 'vgg':
            self._encoder = vgg_encoder.VGG16Encoder(phase=phase)
        elif net_flag == 'dense':
            self._encoder = dense_encoder.DenseEncoder(phase=phase, L=20, N=5,
                                                       growth_rate=8, with_bc=True)
        self._decoder = fcn_decoder.FCNDecoder()

    def build_model(self, input_tensor, name):
        """
        build forward propagation model

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # first encode
            encode_ret = self._encoder.encode(input_tensor=input_tensor, name='encode')

            # second decode
            if self._net_flag.lower() == 'vgg':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                   name='decode',
                                                   decode_layer_list=['pool5',
                                                                      'pool4',
                                                                      'pool3'])
                return decode_ret
            elif self._net_flag.lower() == 'dense':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['Dense_Block_5',
                                                                     'Dense_Block_4',
                                                                     'Dense_Block_3'])
                return decode_ret

    def compute_loss(self, input_tensor, label, name):
        """

        :param input_tensor:
        :param label:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # build model and get logits
            inference_ret = self.build_model(input_tensor=input_tensor, name='inference')
            # compute loss
            decode_logits = inference_ret['logits']
            # add bounded inverse class weights
            inverse_class_weights = tf.divide(1.0,
                                              tf.log(tf.add(tf.constant(1.02, dtype=tf.float32))),
                                              tf.nn.softmax(decode_logits))
            decode_logits_weighted = tf.multiply(decode_logits, inverse_class_weights)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decode_logits_weighted,
                                                                  labels=tf.squeeze(label, squeeze_dims=[3]),
                                                                  name='entropy_loss')

            ret = dict()
            ret['entropy_loss'] = loss
            ret['inference_logits'] = inference_ret['logits']

            return ret