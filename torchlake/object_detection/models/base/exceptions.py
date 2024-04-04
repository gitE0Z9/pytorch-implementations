class BackboneBuildFailure(Exception):

    def __init__(self, name: str = "") -> None:
        super().__init__("Build backbone %s failed, please check backbone" % (name,))
