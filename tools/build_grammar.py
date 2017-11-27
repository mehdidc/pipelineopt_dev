import logging
from collections import OrderedDict
import numpy as np
from tpot.config_classifier import classifier_config_dict

import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from grammaropt.grammar import extract_rules_from_grammar
from grammaropt.grammar import build_grammar
from grammaropt.types import Int
from grammaropt.types import Float

from autosklearn.pipeline.components import SimpleClassificationPipeline

log = logging.getLogger(__name__)
hndl = logging.StreamHandler()
log.addHandler(hndl)


rules_tpl = r"""pipeline = "make_pipeline" op elements cm estimator cp
elements = (preprocessor cm elements) / preprocessor
preprocessor = {preprocessors} 
estimator = {estimators}
{body}
op = "("
cp = ")"
cm = ","
eq = "="
bool = "True" / "False"
none = "None"
"""

def _ordered(d):
    dout = OrderedDict()
    keys = sorted(d.keys())
    for k in keys:
        dout[k] = d[k]
    return dout
classifier_config_dict = _ordered(classifier_config_dict)

def _val_to_str(val):
    return "\"{}\"".format(val)
    replace = dict(zip('0123456789', 'ijklmnopqr'))
    replace['.'] = 's'
    replace['e'] = 't'
    replace['-'] = 'u'
    s = str(val)
    s = [replace[c] for c in s]
    return ''.join(s)

def _sort_func(k):
    return "0" * (10-len(k)) + k

def _slug(s):
    return s.lower().replace('.', '_')


def generate_grammar():
    clf = SimpleClassificationPipeline()
    space = clf._get_hyperparameter_search_space()
    print(space)


if __name__ == '__main__':
    generate_grammar() 
