from enum import IntEnum, Enum


class BaseIntEnum(IntEnum):

    def __repr__(self):
        return self.__class__.__name__ + '.{}'.format(self.name)


class BaseEnum(Enum):

    def __repr__(self):
        return self.__class__.__name__ + '.{}'.format(self.name)
