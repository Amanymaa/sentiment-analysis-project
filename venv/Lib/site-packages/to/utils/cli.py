from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.validation import Validator, ValidationError
from fuzzyfinder import fuzzyfinder
from pygments.token import Token
import shlex as lex
import os
import re
from .helpers import *


class CommandValidator(Validator):

    def __init__(self, trainer):
        self.trainer = trainer
        self.commands = self.trainer.commands

    def validate(self, document):
        text = document.text

        parts = list(filter(None, text.split(' ')))
        command, length = parts[0], len(parts)

        if command == 'list' or command == 'help':
            if length != 1:
                raise ValidationError(message='{}: No argument required for this command.'.format(command))
        elif command == 'use':
            if length != 2:
                raise ValidationError(message='use: A valid configuration file is required.')
            elif not self.trainer.has_cfg(parts[1]):
                raise ValidationError(message='use: Configuration "{}" doesn\'t exist.'.format(parts[1]))
        elif command == 'load':
            if length == 1:
                pass
            elif length != 2 or not parts[1].isdigit():
                raise ValidationError(message='load: Please enter (and only enter) a number for which epoch to load.')
            elif int(parts[1]) != 0 and not self.trainer.has_version(int(parts[1])):
                raise ValidationError(
                    message='load: Epoch {} doesn\'t exist for configuration "{}".'.
                    format(parts[1], self.trainer.current_cfg)
                )
        elif command == 'run':
            if length == 2 and not parts[1].isdigit():
                raise ValidationError(message='run: Please enter a number for number of epochs.')
            elif length > 2:
                raise ValidationError(message='run: Too many parameters.')
        elif command == 'set':
            parts = list(filter(None, text.split(' ', 2)))
            if length != 3:
                raise ValidationError(message='set: This command takes exactly three parameters.')
            elif parts[1] == 'learning_rate' and not is_num(parts[2]):
                raise ValidationError(message='set: Learning rate can only be a number, not "{}".'.format(parts[2]))
            elif parts[1] == 'name' or parts[1] == 'config_name':
                raise ValidationError(message='set: Setting parameter "{}" is prohibited.'.format(parts[1]))

            attrs = list(filter(lambda x: not x.startswith('__'), dir(self.trainer.cfg)))
            if parts[1] not in attrs:
                raise ValidationError(message='set: Parameter "{}" not found in configuration.'.format(parts[1]))
        elif command == 'test' or command == 'validate':
            if length == 2:
                ranges = parts[1].split(':')

                if len(ranges) <= 3:
                    if not all([r.isdigit() for r in ranges]):
                        raise ValidationError(message='{}: Please enter a range for the command.'.format(command))
                else:
                    raise ValidationError(message='run: Too many parameters.')
            elif length > 2:
                raise ValidationError(message='run: Too many parameters.')
        elif command == 'vote':
            parts = list(filter(None, text.split(' ', 1)))

            if len(parts) == 2:
                files = lex.split(parts[1])

                if len(files) == 1:
                    raise ValidationError(message='vote: Enter path for at least two files to vote.')

                options = match_prefix(folder='{}/'.format(self.trainer.submissions_folder), suffix='.csv')
                if len(options) == 0:
                    raise ValidationError(message='vote: No submission files to vote for.')

                for f in files:
                    if f not in options:
                        raise ValidationError(message='vote: Submission file "{}" not found.'.format(f))
        elif command == 'exit':
            pass
        else:
            raise ValidationError(message='{}: Invalid command.'.format(command))


class CommandCompleter(Completer):

    def __init__(self, trainer):
        self.trainer = trainer
        self.commands = self.trainer.commands

    def get_options_for_csv(self, word=None):
        return match_prefix(word, folder='{}/'.format(self.trainer.submissions_folder), suffix='.csv')

    def get_options_for_configs(self, word=None):
        return match_prefix(word, folder='{}/'.format(self.trainer.cfg_folder), suffix='.py')

    def filter_versions(self, prefix=None, greater_than=0):
        epoch, path, paths, versions = self.trainer.get_versions()

        try:
            greater_than = int(greater_than)
        except Exception as e:
            greater_than = 0

        results = [(v, p) for v, p in sorted(zip(versions, paths), key=lambda pair: pair[0])]

        if prefix is not None and (prefix != '' or greater_than != 0):
            filtered = []
            for pair in results:
                if str(pair[0]).startswith(prefix) and pair[0] > greater_than:
                    filtered.append(pair)
            results = filtered

        return results

    def get_completions(self, document, complete_event):
        valid_words = []
        current_line = document.current_line_before_cursor
        word_before_cursor = document.get_word_before_cursor(WORD=True)
        words_this_line = current_line.split(' ')

        if len(words_this_line) <= 1:
            valid_words += self.commands
        else:
            command = words_this_line[0]
            if command == 'run':
                valid_words += [str(i) for i in range(1, 101)]
            elif command == 'load':
                models = self.filter_versions(prefix=word_before_cursor)
                valid_words += [str(m[0]) for m in models]
            elif command == 'use':
                valid_words += self.get_options_for_configs(word_before_cursor)
            elif command == 'set':
                if len(words_this_line) == 2:
                    attrs = list(filter(lambda x: not x.startswith('__'), dir(self.trainer.cfg)))
                    valid_words += attrs
            elif command == 'test' or command == 'validate':
                words = word_before_cursor.split(':')
                length = len(words)

                if length == 1:
                    models = self.filter_versions(prefix=words[0])
                    valid_words += [str(m[0]) for m in models]
                elif length == 2:
                    try:
                        models = self.filter_versions(prefix=words[1], greater_than=words[0])
                        if len(models):
                            prev = int(words[0])
                            curr = int(models[-1][0])
                            valid_words += ['{}:{}'.format(words[0], i) for i in range(prev + 1, curr + 2)]
                    except Exception as e:
                        raise e
                elif length == 3:
                    try:
                        prev = int(words[0])
                        curr = int(words[1])
                        valid_words += ['{}:{}:{}'.format(words[0], words[1], i) for i in range(1, curr - prev + 1)]
                    except Exception as e:
                        raise e
            elif command == 'vote':
                checklist = lex.split(current_line)
                options = list(filter(lambda x: x not in checklist, self.get_options_for_csv(word_before_cursor)))
                valid_words += ['"{}"'.format(o) if ' ' in o else o for o in options]

        matches = fuzzyfinder(word_before_cursor, valid_words)
        for m in matches:
            yield Completion(m, start_position=-len(word_before_cursor))
