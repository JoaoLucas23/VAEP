import d6tflow as d6t
from loaders import WyscoutToSPADL


class wyVAEP(d6t.tasks.TaskCSVPandas):
    competition_name = d6t.Parameter()

    persist = ['teams', 'players', 'actions']

    def requires(self):
        return WyscoutToSPADL(competition_name=self.competition_name)

    def run(self):
        #teams = self.input()['teams'].load()
        #players = self.input()['players'].load()
        actions = self.inputLoad()

        self.save(actions)
        #self.save({'teams':teams, 'players':players, 'actions':actions})

