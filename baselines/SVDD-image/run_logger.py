class NoOpRunLogger:
    def init(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None

    def Image(self, *args, **kwargs):
        return None


RUN_LOGGER = NoOpRunLogger()
