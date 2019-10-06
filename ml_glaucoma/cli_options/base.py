from abc import abstractmethod


class Configurable(object):
    description = None

    def __init__(self, **children):
        self._children = children
        for v in children.values():
            assert v is None or isinstance(v, Configurable)

    def fill(self, parser):
        for child in self._children.values():
            if child is not None:
                child.fill(parser)
        self.fill_self(parser)

    def build(self, **kwargs):
        children_values = {
            k: None if child is None else child.build(**kwargs)
            for k, child in self._children.items()}
        kwargs.update(children_values)
        return self.build_self(**kwargs)

    @abstractmethod
    def fill_self(self, parser):
        raise NotImplementedError

    @abstractmethod
    def build_self(self, **kwargs):
        raise NotImplementedError

    def map(self, fn):
        return MappedConfigurable(self, fn)


class MappedConfigurable(Configurable):
    def __init__(self, base, fn):
        super(MappedConfigurable, self).__init__(base=base)
        self.fn = fn

    def fill_self(self, **kwargs):
        pass

    def build_self(self, base, **kwargs):
        return self.fn(base)
