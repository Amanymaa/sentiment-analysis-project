"""Cli tests."""
import unittest
from os import utime
from pathlib import Path

from clean import cli
from clean import config
from clean.pathutil import rm_recursive

from click.testing import CliRunner


class TestCli(unittest.TestCase):
    """Test cli commands."""

    def setUp(self):
        """Setup cli test."""
        current_dir = Path(__file__).parent.resolve()
        test_cleanrc_path = current_dir / '.cleanrc'
        cleanrc_template_path = current_dir / '.cleanrc.template'
        with test_cleanrc_path.open('w', encoding='utf_8') as f:
            with cleanrc_template_path.open('r', encoding='utf_8') as template:
                f.write(template.read())
        config.config_file_name = test_cleanrc_path
        self.test_cleanrc_path = test_cleanrc_path
        self.current_dir = current_dir
        self.env = {
            'CLEANRC_PATH': str(self.test_cleanrc_path)
        }  # type: dict[str, str]
        self.initialPath = {
            "glob": "fuga",
            "path": "hoge",
            "use_meta_tag": False
        }

    def test_add_config(self):
        """Test add command."""
        runner = CliRunner()
        runner.invoke(cli.add, ['hogehoge', 'fugafuga'], env=self.env)
        c = config.Config(config_path=self.test_cleanrc_path)
        path_list = c.list_glob_path()
        self.assertIn({
            "glob": "hogehoge",
            "path": "fugafuga",
            "use_meta_tag": True
        }, path_list)

    def test_add_same_config(self):
        """Cannot add same path config to a cleanrc."""
        c = config.Config(config_path=self.test_cleanrc_path)
        path_list = c.list_glob_path()
        self.assertEqual(path_list.count(self.initialPath), 1)

        runner = CliRunner()
        runner.invoke(cli.add, ['fuga', 'hoge'], env=self.env)

        c = config.Config(config_path=self.test_cleanrc_path)
        path_list = c.list_glob_path()
        self.assertEqual(path_list.count(self.initialPath), 1)

    def test_list_config(self):
        """List all configs."""
        runner = CliRunner()
        result = runner.invoke(cli.list, env=self.env)
        self.assertEqual("[0] fuga => hoge \n", result.output)

    def test_delete_config(self):
        """Delete a config."""
        test_value = {
            "glob": "fuga",
            "path": "hoge",
            "use_meta_tag": False
        }  # type: dict[str, str]
        c = config.Config(config_path=self.test_cleanrc_path)  # type: Config
        path_list = c.list_glob_path()  # type: list[dict[str, str]]
        index = path_list.index(test_value)
        runner = CliRunner()
        runner.invoke(cli.delete, [str(index)], env=self.env)
        c = config.Config(config_path=self.test_cleanrc_path)
        path_list = c.list_glob_path()
        self.assertNotIn(test_value, path_list)

    def test_move(self):
        """Run cleaning."""
        test_from_dir = self.current_dir / 'test_from_dir'  # type: Path
        test_to_dir = self.current_dir / 'test_to_dir'  # type: Path
        # Remove directory and remake
        rm_recursive(test_from_dir)
        test_from_dir.mkdir()
        rm_recursive(test_to_dir)
        # Initialize test files
        test_file_names = {'foo', 'bar'}
        for name in test_file_names:
            (test_from_dir / name).touch()
        test_value = {
            "glob": str(test_from_dir) + '/*',
            "path": str(test_to_dir)
        }
        runner = CliRunner()
        runner.invoke(
            cli.add, [test_value['glob'], test_value['path']], env=self.env)
        runner.invoke(cli.run, env=self.env)
        files = test_to_dir.glob('*')
        self.assertEqual(test_file_names, {str(x.name) for x in files})

    def test_meta_tag(self):
        """Use meta tag."""
        test_from_dir = self.current_dir / 'test_from_dir'  # type: Path
        test_to_dir = self.current_dir / 'test_to_dir'
        rm_recursive(test_from_dir)
        test_from_dir.mkdir()
        rm_recursive(test_to_dir)
        # Create test files
        test_files = [(1520780400, 'foo'), (1521039600, 'bar')]
        for data in test_files:
            file_path = test_from_dir / data[1]
            file_path.touch()
            # Set file update time
            utime(str(file_path), (data[0], data[0]))
        cleanrc = ('''
                {{"path": [
                    {{"path": "{}", "glob": "{}", "use_meta_tag": true}}
                ]}}''').format(
            str(test_to_dir) + '/<YEAR><MONTH><DAY><HOUR><MINUTE><SECOND>',
            str(test_from_dir) + '/*')
        with self.test_cleanrc_path.open('w', encoding="utf_8") as f:
            f.write(cleanrc)
        # Run!!!!
        runner = CliRunner()
        result = runner.invoke(cli.run, env=self.env)
        files = [x for x in test_to_dir.glob('**/*') if x.is_file()]
        self.assertEqual({
            str(test_to_dir) + '/20180312000000/foo',
            str(test_to_dir) + '/20180315000000/bar'
        }, {str(x)
            for x in files})


if __name__ == '__main__':
    unittest.main()
