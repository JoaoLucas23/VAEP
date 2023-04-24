import d6tflow as d6t

from features import CreateVAEPFeatures
from wyVAEP import wyVAEP
from computeVAEP import ComputeVAEP

COMPETITION_NAME = 'English first division'
TRAIN_COMPETITIONS = ['Spanish first division', 'Italian first division', 'French first division']

wy = d6t.Workflow(ComputeVAEP, params={'competition_name': COMPETITION_NAME, 'train_competitions': TRAIN_COMPETITIONS})
wy.run()
out = wy.outputLoad()
print("Average VAEP per action: {}".format(out['VAEP'].mean()))

