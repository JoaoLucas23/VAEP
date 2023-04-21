import d6tflow as d6t

from wyVAEP import wyVAEP
from loaders import WyscoutToSPADL

wy = WyscoutToSPADL(competition_name='English first division')
wy.run()
out = wy.outputLoad()

print(out)