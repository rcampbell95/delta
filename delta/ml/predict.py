import numpy as np

import tensorflow as tf

from delta.imagery import rectangle

#pylint: disable=unsubscriptable-object
# Pylint was barfing lines 32 and 76. See relevant bug report
# https://github.com/PyCQA/pylint/issues/1498

def predict_array(model, data):

    net_input_shape = model.get_input_shape_at(0)[1:]
    net_output_shape = model.get_output_shape_at(0)[1:]

    assert net_input_shape[2] == data.shape[2],\
           'Model expects %d input channels, data has %d channels' % (net_input_shape[2], data.shape[2])

    out_shape = (data.shape[0] - net_input_shape[0] + net_output_shape[0],
                 data.shape[1] - net_input_shape[1] + net_output_shape[1])
    image = tf.convert_to_tensor(data)
    image = tf.expand_dims(image, 0)
    # TODO: Change stride so not constantly overlapping chunks for AE.
    chunks = tf.image.extract_patches(image, [1, net_input_shape[0], net_input_shape[1], 1],
                                      [1, net_output_shape[0], net_output_shape[1], 1],
                                      [1, 1, 1, 1], padding='VALID')
    chunks = tf.reshape(chunks, (-1,) + net_input_shape)

    # TODO: Figure out correct shape for best for AE.
    best = np.zeros((chunks.shape[0],) + net_output_shape, dtype=np.float32)
    ## TODO: other data types, configurable batch size
    BATCH_SIZE=1000
    for i in range(0, chunks.shape[0], BATCH_SIZE):
        # batches = model.predict_on_batch(chunks[i:i+BATCH_SIZE])
        best[i:i+BATCH_SIZE] = model.predict_on_batch(chunks[i:i+BATCH_SIZE])

    retval = np.zeros(out_shape + (net_output_shape[-1],))
    for chunk_idx in range(0, best.shape[0]):
        r = (chunk_idx // (  out_shape[1] // net_output_shape[1])) * net_output_shape[0]
        c = (chunk_idx  % ( out_shape[1] // net_output_shape[1])) * net_output_shape[1]
        retval[r:r+net_output_shape[0],c:c+net_output_shape[1],:] = best[chunk_idx,:,:,:]

    return retval

def predict_validate(model, image, label, input_bounds=None, probabilities=False, show_progress=False):
    """Like predict but returns (predicted image, error image, percent correct)."""
    net_input_shape = model.get_input_shape_at(0)[1:]
    net_output_shape = model.get_output_shape_at(0)[1:]
    offset_r = - net_input_shape[0] + net_output_shape[0]
    offset_c = - net_input_shape[1] + net_output_shape[1]
    block_size_x = net_input_shape[0] * (256 // net_input_shape[0])
    block_size_y = net_input_shape[1] * (256 // net_input_shape[1])

    # Set up the output image
    if not input_bounds:
        input_bounds = rectangle.Rectangle(0, 0, width=image.width(), height=image.height())

    if not probabilities:
        result = np.zeros((input_bounds.width() + offset_r, input_bounds.height() + offset_c), dtype=np.uint8)
    else:
        result = np.zeros((input_bounds.width() + offset_r,
                           input_bounds.height() + offset_c, net_output_shape[-1]), dtype=np.float32)
    errors = None
    confusion_matrix = None
    if label:
        errors = np.zeros((input_bounds.width() + offset_r, input_bounds.height() + offset_c), dtype=np.bool)
        confusion_matrix = np.zeros((net_output_shape[-1], net_output_shape[-1]), dtype=np.int32)

    def callback_function(roi, data):
        pred_image = predict_array(model, data)

        block_x = (roi.min_x - input_bounds.min_x)
        block_y = (roi.min_y - input_bounds.min_y)
        (sx, sy) = (block_x , block_y)

        if probabilities:
            result[sx : sx + pred_image.shape[0], sy : sy + pred_image.shape[1], :] = pred_image
        pred_image = np.argmax(pred_image, axis=2)

        if not probabilities:
            result[sx : sx + pred_image.shape[0], sy : sy + pred_image.shape[1]] = pred_image

        if label:
            start_x = roi.min_x + (roi.width() - pred_image.shape[0]) // 2
            start_y = roi.min_y + (roi.height() - pred_image.shape[1]) // 2
            label_roi = rectangle.Rectangle(start_x, start_y,
                                            start_x + pred_image.shape[0], start_y + pred_image.shape[1])
            labels = np.squeeze(label.read(label_roi))
            errors[sx : sx + pred_image.shape[0], sy : sy + pred_image.shape[1]] = labels != pred_image
            cm = tf.math.confusion_matrix(np.ndarray.flatten(labels),
                                          np.ndarray.flatten(pred_image),
                                          net_output_shape[-1])
            confusion_matrix[:, :] += cm

    # TODO think about the below to test.
    output_rois = input_bounds.make_tile_rois(block_size_x - offset_r, block_size_y - offset_c,
                                              include_partials=False, overlap_amount=-offset_r)

    image.process_rois(output_rois, callback_function, show_progress=show_progress)
    return (result, errors, confusion_matrix)

# TODO: Save data as we progress through it
def predict(model, image, input_bounds=None, probabilities=False, show_progress=False):
    """Returns the predicted image given a model, chunk size, and image."""
    return predict_validate(model, image, None, input_bounds, probabilities, show_progress)[0]
