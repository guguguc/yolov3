import tensorflow as tf


def enable_mem_group():
    gpu = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)


def debugging(root_path):
    tf.debugging.experimental.enable_dump_debug_info(
        dump_root='logs/debugging',
        tensor_debug_mode='FULL_HEALTH',
        circular_buffer_size=-1)


def mix_precision():
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
