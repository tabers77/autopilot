import datetime as dt
import math
import unittest

import pandas as pd

from taberspilotml.pre_modelling import encoders as enc


class GetEncodedWrapperTestCase(unittest.TestCase):

    def test_raisesexception_withnulls(self):
        with self.assertRaises(ValueError) as ve:
            enc.default_encoding(pd.DataFrame(data=[(None,)], columns=['a']))

        self.assertEqual('There are missing values in your dataset', ve.exception.args[0])

    def test_encodenulls__none_nan(self):
        encoded = enc.default_encoding(
            pd.DataFrame(data=[(None, math.nan),
                               (1, 'a')], columns=['a', 'b']), encode_nulls=True)

        pd.testing.assert_frame_equal(pd.DataFrame(data=[(0, 0),
                                                         (1, 1)],
                                                   columns=['encoded_nulls_a', 'encoded_nulls_b']), encoded)

    def test_encodenulls__nan_none(self):
        encoded = enc.default_encoding(
                pd.DataFrame(data=[(math.nan, None),
                                   (1, 'a')], columns=['a', 'b']), encode_nulls=True)

        pd.testing.assert_frame_equal(pd.DataFrame(data=[(0, 0),
                                                         (1, 1)],  # Is this really desired?
                                                   columns=['encoded_nulls_a', 'encoded_nulls_b']), encoded)

    def test_categoryasint_encoding__binarycase(self):
        df = pd.DataFrame(data=[('a', 'yes'),
                                ('b', 'no'),
                                ('a', 'no')], columns=['Cat1', 'Cat2'])

        encoded = enc.default_encoding(df)
        pd.testing.assert_frame_equal(pd.DataFrame(
            data=[(0, 1),
                  (1, 0),
                  (0, 0)], columns=['Cat1', 'Cat2']), encoded)

    def test_categoryasint_encoding__multiclass_case(self):
        df = pd.DataFrame(data=[('a', 'yes'),
                                ('b', 'no'),
                                ('a', 'no'),
                                ('c', 'maybe')], columns=['Cat1', 'Cat2'])

        encoded = enc.default_encoding(df)
        pd.testing.assert_frame_equal(pd.DataFrame(
            data=[(0, 2),
                  (1, 1),
                  (0, 1),
                  (2, 0)], columns=['Cat1', 'Cat2']), encoded)

    def test_date_expanded(self):
        df = pd.DataFrame(data=[(0, dt.datetime(year=2021, day=5, month=10)),
                                (1, dt.datetime(year=2021, day=5, month=11))], columns=['a', 'date'])

        encoded = enc.default_encoding(df)
        pd.testing.assert_frame_equal(pd.DataFrame(
            data=[(0, 5, 10, 2021, 1),
                  (1, 5, 11, 2021, 4)],
            columns=['a', 'date_day', 'date_month', 'date_year', 'date_weekday']), encoded)


if __name__ == '__main__':
    unittest.main()
