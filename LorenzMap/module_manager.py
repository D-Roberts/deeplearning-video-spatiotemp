
class ModuleManager(object):
    def __init__(self):
        pass

    def build(self):
        """

        :return:
        """
        net = Lorenz(L=self.dilation_depth, in_channels=self.in_channels, k=2, M=1)
        net.collect_params().initialize(mx.init.Xavier(magnitude=2, rnd_type='gaussian', factor_type='in'),
                                        ctx=self.ctx)
        return net

