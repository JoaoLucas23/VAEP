import d6tflow as d6t

from wyVAEP import wyVAEP

wy = wyVAEP(competition_name='English first division')
wy.run()
out = wy.outputLoad()

print(out)