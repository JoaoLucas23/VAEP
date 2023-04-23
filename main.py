import d6tflow as d6t

from wyVAEP import wyVAEP

COMPETITION_NAME = 'English first division'

wy = d6t.Workflow(wyVAEP, params={'competition_name': COMPETITION_NAME})
wy.run()
out = wy.outputLoad()
