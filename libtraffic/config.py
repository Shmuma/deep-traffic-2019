import configparser


class Settings:
    def __init__(self, file_name):
        self.parser = configparser.ConfigParser()
        self.parser.read([file_name])

    @property
    def env_lanes_side(self):
        return self.parser.getint('env', 'lanes_side', fallback=3)

    @property
    def env_patches_ahead(self):
        return self.parser.getint('env', 'patches_ahead', fallback=20)

    @property
    def env_patches_behind(self):
        return self.parser.getint('env', 'patches_behind', fallback=10)

    @property
    def env_history(self):
        return self.parser.getint('env', 'history', fallback=3)

    @property
    def env_steps_limit(self):
        return self.parser.getint('env', 'steps_limit', fallback=100)

    @property
    def train_gamma(self):
        return self.parser.getfloat('train', 'gamma', fallback=0.95)

    @property
    def train_lr(self):
        return self.parser.getfloat('train', 'lr', fallback=1e-4)

    @property
    def train_replay_size(self):
        return self.parser.getint('train', 'replay_size', fallback=100000)

    @property
    def train_replay_initial(self):
        return self.parser.getint('train', 'replay_initial', fallback=1000)

    @property
    def train_cuda(self):
        return self.parser.getboolean('train', 'cuda')

    @property
    def train_eps_start(self):
        return self.parser.getfloat('train', 'eps_start', fallback=1.0)

    @property
    def train_eps_end(self):
        return self.parser.getfloat('train', 'eps_end', fallback=0.05)

    @property
    def train_eps_steps(self):
        return self.parser.getint('train', 'eps_steps')

    @property
    def train_batch_size(self):
        return self.parser.getint('train', 'batch_size', fallback=256)

    @property
    def train_l2_reg(self):
        return self.parser.getfloat('train', 'l2_reg', fallback=0.0)

    @property
    def train_add_steps_limit_slope(self):
        return self.parser.getfloat('train', 'add_steps_limit_slope', fallback=0.0)

    @property
    def train_add_steps_limit_max(self):
        return self.parser.getint('train', 'add_steps_limit_max', fallback=0)

    @property
    def train_net_sync_steps(self):
        return self.parser.getint('train', 'net_sync_steps')

    @property
    def train_test_steps(self):
        return self.parser.getint('train', 'test_steps', fallback=1000)

    @property
    def train_test_interval(self):
        return self.parser.getint('train', 'test_interval', fallback=1000)

    @property
    def train_test_rounds(self):
        return self.parser.getint('train', 'test_rounds', fallback=5)
