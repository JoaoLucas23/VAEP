import d6tflow as d6t
from loaders import WyscoutToSPADL


class wyVAEP(d6t.tasks.TaskCSVPandas):
    competition_name = d6t.Parameter()

    def requires(self):
        return WyscoutToSPADL(competition_name=self.competition_name)

    def run(self):
        teams = self.inputLoad('teams')
        players = self.inputLoad('players')
        actions = self.inputLoad('actions')

        self.save(teams)
        self.save(players)
        self.save(actions)
