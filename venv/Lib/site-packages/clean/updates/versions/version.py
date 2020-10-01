"""Version management class."""
from abc import ABCMeta
from abc import abstractmethod


class Version(metaclass=ABCMeta):
    """Version migrating class."""

    @abstractmethod
    def up(self, config: dict):
        """Upgrade config file."""
        pass

    @abstractmethod
    def down(self, config: dict):
        """Downgrade config file."""
        pass
