# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.utils import check_array


class ColumnTransformer(BaseEstimator, TransformerMixin):
    """Wrapper around sklearn.compose.ColumnTransformer"""
    def __init__(self, transformer, cols, remainder):
        self.transformer = transformer
        self.cols = cols
        self.remainder = remainder

    def fit(self, X, y=None):
        valid_cols = [n for n in cols if n < X.shape[1]]
        self.col_transformer = make_column_transformer((self.transformer, valid_cols), remainder=self.remainder)
        self.col_transformer.fit(X, y)
        return self

    def transform(self, X):
        return self.col_transformer.transform(X)

