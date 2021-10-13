import unittest

import tensorflow as tf

from tf_msgms import gms, msgms


class MSGMSTestCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        image_string = tf.io.read_file('color-blind-test.png')
        tf_img = tf.cast(tf.image.decode_png(image_string, channels=3), dtype=tf.float32)
        cls.tf_img = tf.expand_dims(tf_img, axis=0)  # B, H, W, C
        cls.diff_img = tf.expand_dims(tf.image.random_contrast(tf_img, 0.2, 0.5), axis=0)

    def test_gms_same_image(self):
        loss_score = gms(self.tf_img, self.tf_img)
        print(f'{loss_score=}')
        self.assertEqual(loss_score, 0)

    def test_msgms_same_image(self):
        loss_score = msgms(self.tf_img, self.tf_img)
        print(f'{loss_score=}')
        self.assertEqual(loss_score, 0)

    def test_gms_diff_image(self):
        loss_score = gms(self.tf_img, self.diff_img)
        print(f'{loss_score=}')
        self.assertNotEqual(loss_score, 0)

    def test_msgms_diff_image(self):
        loss_score = msgms(self.tf_img, self.diff_img)
        print(f'{loss_score=}')
        self.assertNotEqual(loss_score, 0)


if __name__ == '__main__':
    unittest.main()
