import d6tflow as d6t

from wyVAEP import wyVAEP

wy = wyVAEP(competition_name='Premier League')
wy.run()
out = wy.outputLoad()

print(out)