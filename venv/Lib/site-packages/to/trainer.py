import re
import os
import traceback
import importlib.util

from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.validation import ValidationError
from colored import fg, bg, attr

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .utils.cli import *
from .utils.helpers import *
from .utils.batch_logger import *
from .utils.options import *
from .net import *
from .data.dataset import *


class Trainer(object):

    #----------------------------------------------------------------------------------------------------------
    # Initialization
    #----------------------------------------------------------------------------------------------------------

    def __init__(self):
        super(Trainer, self).__init__()

        self.epoch_ran = 0
        self.logger = Logger(self)
        self.name = sys.argv[0].replace('.py', '')
        self.commands = ['list', 'help', 'use', 'load', 'run', 'test', 'validate', 'set', 'exit']
        # Configurations
        self.cfg_folder = 'configurations'
        self.default_cfg = 'default'
        self.current_cfg = self.default_cfg
        self.current_cfg_path = None
        # Models
        self.models_folder = 'models'
        self.Model = NeuralNetwork
        self.cuda_enabled = False
        # Events
        self.event_handlers = {}
        # Data
        self.DataLoader = None
        self.DataSet = DataSet
        # Submission
        self.submissions_folder = 'submissions'

        if len(sys.argv) == 2:
            self.load_cfg('{}/{}.py'.format(self.cfg_folder, sys.argv[1].replace('.py', '')))
        else:
            self.load_cfg('{}/{}.py'.format(self.cfg_folder, self.current_cfg))
        self.reset()

    def reset(self):
        self.__init_model()
        self.__init_optim()
        self.__init_loss_fn()
        return self

    def __init_folder(self):
        self.cfg_folder = get(self.cfg, TrainerOptions.CFG_FOLDER.value, default='configurations')
        self.models_folder = get(self.cfg, TrainerOptions.MODELS_FOLDER.value, default='models')
        self.submissions_folder = get(self.cfg, TrainerOptions.SUBMISSIONS_FOLDER.value, default='submissions')

    def __init_model(self):
        self.model = self.Model(self.cfg)
        init_model_parameters(self.model)
        if torch.cuda.is_available():
            self.cuda_enabled = True
            self.model = self.model.cuda()

    def __init_optim(self):
        Optimizer = get(self.cfg, TrainerOptions.OPTIMIZER.value, default=optim.Adam)
        optim_args = get(self.cfg, TrainerOptions.OPTIMIZER_ARGS.value, default={'lr': 0.01})
        self.optimizer = Optimizer(self.model.parameters(), **optim_args)

        Scheduler = get(self.cfg, TrainerOptions.SCHEDULER.value, default=None)
        sched_args = get(self.cfg, TrainerOptions.SCHEDULER_ARGS.value, default=[])
        sched_kwargs = get(self.cfg, TrainerOptions.SCHEDULER_KWARGS.value, default={})
        self.scheduler = None
        if Scheduler is not None:
            self.scheduler = Scheduler(self.optimizer, *sched_args, **sched_kwargs)

    def __init_loss_fn(self):
        Fn = get(self.cfg, TrainerOptions.LOSS_FN.value, default=nn.CrossEntropyLoss)
        self.loss_fn = Fn()

    #----------------------------------------------------------------------------------------------------------
    # Folder
    #----------------------------------------------------------------------------------------------------------

    def set_models_folder(self, models_folder):
        self.models_folder = models_folder
        return self

    def set_submissions_folder(self, submissions_folder):
        self.submissions_folder = submissions_folder
        return self

    def set_configurations_folder(self, cfg_folder):
        self.cfg_folder = cfg_folder
        self.load('{}/{}.py'.format(self.cfg_folder, self.current_cfg))
        self.reset()
        return self

    #----------------------------------------------------------------------------------------------------------
    # Configuration
    #----------------------------------------------------------------------------------------------------------

    def has_cfg(self, cfg):
        if not cfg.endswith('.py'):
            cfg += '.py'
        if '/' not in cfg:
            cfg = os.path.join(csd(), self.cfg_folder, cfg)
        return os.path.isfile(cfg)

    def load_cfg(self, cfg_file):
        if not cfg_file.startswith(self.cfg_folder):
            cfg_file = os.path.join(self.cfg_folder, cfg_file)
        if not cfg_file.endswith('.py'):
            cfg_file += '.py'
        path = os.path.join(csd(), cfg_file)

        try:
            p('Loading configuration file at "{}"'.format(path))
            spec = importlib.util.spec_from_file_location('configuration', path)
            self.cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.cfg)

            self.current_cfg = filename(path).replace('.py', '')
            self.current_cfg_path = path

            self.__init_folder()
        except IOError as e:
            raise Exception('Configuration file not found at "{}".'.format(path))

        return self

    #----------------------------------------------------------------------------------------------------------
    # Events
    #----------------------------------------------------------------------------------------------------------

    def bind(self, event, handler):
        if isinstance(event, TrainerEvents):
            self.event_handlers[event.value] = handler
        else:
            raise Exception('Event "{}" should be a TrainerEvents.'.format(event))

        return self

    #----------------------------------------------------------------------------------------------------------
    # DataSet and DataLoader
    #----------------------------------------------------------------------------------------------------------

    def set_dataloader(self, DataLoader):
        self.DataLoader = DataLoader
        return self

    def set_dataset(self, DataSet):
        self.DataSet = DataSet
        return self

    def __get_dataloader(self, data_type, debug=True):
        if self.DataLoader is not None:
            return self.DataLoader(self.cfg, data_type)
        else:
            dataset = self.DataSet(self.cfg, data_type, debug)

            if has(self.event_handlers, TrainerEvents.CUSTOMIZE_DATALOADER.value):
                return get(self.event_handlers, TrainerEvents.CUSTOMIZE_DATALOADER.value)(self.cfg, data_type, dataset)
            else:
                should_shuffle = data_type != TEST
                batch_size = get(self.cfg, TrainerOptions.BATCH_SIZE.value, default=64)
                return DataLoader(dataset, batch_size=batch_size, shuffle=should_shuffle)

    #----------------------------------------------------------------------------------------------------------
    # Model
    #----------------------------------------------------------------------------------------------------------

    def set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        self.cfg.learning_rate = new_lr
        if has(self.cfg, TrainerOptions.OPTIMIZER_ARGS.value, 'lr'):
            get(self.cfg, TrainerOptions.OPTIMIZER_ARGS.value)['lr'] = new_lr

        return self

    def get_lr(self):
        lr = [g['lr'] for g in self.optimizer.param_groups]
        return lr

    def set_model(self, Model):
        self.Model = Model
        self.reset()
        return self

    def save_model(self, percentage=None, loss=None):
        mkdirp(os.path.join(csd(), self.models_folder, self.name))

        path = '{}/{}/{} - {:03d}'.format(self.models_folder, self.name, self.current_cfg, self.epoch_ran)
        if percentage is not None:
            path += ' - {:.2f}%'.format(percentage)
        if loss is not None:
            path += ' - {:.6f}'.format(loss)
        path += '.model'
        path = os.path.join(csd(), path)

        p('Saving neural network "{}" using configuration "{}" to disk at "{}"'.format( \
            self.name, self.current_cfg, path))
        torch.save(self.model.state_dict(), path)

        return self

    def load_model(self, epoch=None):
        pattern = None

        if epoch == 0:
            p('Resetting model to primitive state.')
            return self.reset()

        epoch, path, files, versions = self.get_versions(epoch)
        if path is None and epoch is not None:  # Can't find the exact epoch, loading the highest.
            epoch, path, files, versions = self.get_versions()

        if epoch > 0 and path is not None:
            p('Loading neural network "{}" using configuration "{}" and epoch "{}" at "{}"'.format( \
                self.name, self.current_cfg, epoch, path))
            try:
                if torch.cuda.is_available():
                    self.model.load_state_dict(torch.load(path))
                else:
                    self.model.load_state_dict(torch.load(path, lambda storage, loc: storage))

                self.epoch_ran = epoch
            except Exception as e:
                p('Failed to load model at path "{}"'.format(path))
                traceback.print_exc()
        else:
            p('No saved model for neural network "{}" using configuration "{}".'.format(self.name, self.current_cfg))

        return self

    def has_version(self, epoch):
        version, path, files, versions = self.get_versions(epoch)
        return version > 0 and version == epoch

    def get_versions(self, epoch=None):
        folder = csd()

        if epoch is not None:
            pattern = '{}/{}/{} - {:03d}*.model'.format(self.models_folder, self.name, self.current_cfg, epoch)
        else:
            pattern = '{}/{}/{}*.model'.format(self.models_folder, self.name, self.current_cfg)

        files = find_pattern(os.path.join(folder, pattern), relative_to=folder)
        if len(files) > 0:
            versions = [int(re.findall(' \d{3} |$', filename(f))[0]) for f in files]

            epoch = max(versions)
            i = versions.index(epoch)
            path = files[i]

            return epoch, path, files, versions

        return (0, None, files, [])

    #----------------------------------------------------------------------------------------------------------
    # CLI
    #----------------------------------------------------------------------------------------------------------

    def cli(self):
        print()
        print('----------------------------------------------------------')
        print('|                                                        |')
        print('|        Welcome to Flare Neural Network Trainer.        |')
        print('|                                                        |')
        print('----------------------------------------------------------')
        print()

        if get(self.cfg, TrainerOptions.AUTO_RELOAD_SAVED_MODEL.value, default=False):
            self.load_model()

        mkdirp('.flare')
        touch('.flare/history')

        should_exit = False
        while not should_exit:
            c = prompt(
                '> ',
                history=FileHistory('.flare/history'),
                auto_suggest=AutoSuggestFromHistory(),
                completer=CommandCompleter(self),
                validator=CommandValidator(self)
            )
            try:
                should_exit = self.process_command(c)
            except Exception as e:
                traceback.print_exc()

        return self

    def process_command(self, c):
        parts = list(filter(None, c.split(' ')))
        command = parts[0]

        if command == 'list':
            self.list()
        elif command == 'help':
            self.help()
        elif command == 'use':
            self.load_cfg('{}/{}.py'.format(self.cfg_folder, parts[1].replace('.py', '')))
        elif command == 'load':
            if len(parts) == 2:
                self.load_model(int(parts[1]))
            else:
                self.load_model()
        elif command == 'run':
            if len(parts) == 1:
                self.run()
            else:
                self.run(int(parts[1]))
        elif command == 'set':
            parts = list(filter(None, c.split(' ', 2)))
            self.set(parts[1], parts[2])
        elif command == 'test' or command == 'validate':
            fn = self.test if command == 'test' else self.validate

            if len(parts) == 1:
                fn()
            else:
                locs = list(map(int, parts[1].split(':')))
                if len(locs) == 1:
                    if self.load_model(locs[0]):
                        fn()
                    else:
                        p('Skipping test because epoch {} cannot be loaded correctly.'.format(locs[0]))
                else:
                    for i in range(*locs):
                        if self.load_model(i):
                            fn()
                        else:
                            p('Skipping test because epoch {} cannot be loaded correctly.'.format(i))
        elif command == 'exit':
            return True
        return False

    #----------------------------------------------------------------------------------------------------------
    # Commands
    #----------------------------------------------------------------------------------------------------------

    def list(self):
        color = fg(45)
        parameter = fg(119)
        reset = attr('reset')

        def colorize(o):
            return '{}{}{}'.format(color, o, reset)

        configs = [
            ('Module', self.name, color),
            ('Epoch', self.epoch_ran, color),
            ('Configuration', self.current_cfg, color),
            ('Configuration Path', self.current_cfg_path, color),
            ('Configuration Folder', self.cfg_folder, color),
            ('Models Folder', self.models_folder, color),
            ('Submissions Folder', self.submissions_folder, color),
            None,
        ]

        for k in list(filter(lambda x: not x.startswith('__'), dir(self.cfg))):
            v = getattr(self.cfg, k)
            configs.append((k, v, parameter))
        max_key_len = max([len(o[0]) if o else 0 for o in configs])

        for o in configs:
            if o is None:
                print()
            else:
                w('{}{} :    {}'.format(o[0], ' ' * (max_key_len - len(o[0])), o[2]))
                w(re.sub('^    ', ' ' * (max_key_len + 6), ff(o[1], prefix='    '), flags=re.M))
                print(reset)

        return self

    def help(self):
        command = fg(45)
        parameter = fg(119)
        sample = fg(105)
        reset = attr('reset')
        indent = '  '
        print(
            indent + """
            {0}python {1}<PYTHON>{3} {1}[CONFIG]{3} {1}[EPOCH]{3}{3}
            You can to specify the configuration file path and epoch count to load at script
            launch where {1}<PYTHON>{3} is the location of your python file, {1}[CONFIG]{3} is the
            location of your configuration file and {1}[EPOCH]{3} is the epoch count you wish to
            load.
            e.g: {2}python nn.py default 2{3}

            {0}list:{3}
                  Usage: {0}list{3}
                  List current module, epoch count and configuration file path.
                  e.g: {2}list{3}
            {0}help:{3}
                  Usage: {0}help{3}
                  Print help message.
                  e.g: {2}help{3}
            {0}use:{3}
                  Usage: {0}use{3} {1}<PATH>{3}
                  Switch to configuration file located at {1}<PATH>{3}.
                  e.g: {2}use default{3}
            {0}load:{3}
                  Usage: {0}load{3} {1}<EPOCH>{3}
                  Load previously trained model at epoch {1}<EPOCH>{3}.
                  e.g: {2}load 10{3}
            {0}run:{3}
                  Usage: {0}run{3} {1}[COUNT]{3}
                  Run training, optionally {1}[COUNT]{3} times
                  e.g: {2}run{3} OR {2}run 10{3}
            {0}set:{3}
                  Usage: {0}set{3} {1}<ATTR> <VALUE>{3}
                  Set the value in configuration dynamically, this does NOT overwrite the
                  configuration file.
                  e.g: {2}set learn_rate 0.01{3}
            {0}test:{3}
                  Usage: {0}test{3} {1}[EPOCH]{3}
                  Test using the model trained, optionally using at epoch {1}[EPOCH]{3}.
                  {1}[EPOCH]{3} can be a range input to range() or an integer.
                  e.g: {2}test 10{3} OR {2}test 1:10:2{3}
            {0}validate:{3}
                  Usage: {0}validate{3} {1}[EPOCH]{3}
                  Validate using the model trained, optionally using at epoch {1}[EPOCH]{3}.
                  {1}[EPOCH]{3} can be a range input to range() or an integer.
                  e.g: {2}validate 10{3} OR {2}validate 1:10:2{3}
            """.format(command, parameter, sample, reset).replace('\t\t\t', indent).strip()
        )

        return self

    def set(self, key, val):
        p('Setting configuration key "{}" to "{}"'.format(key, val))

        if key == 'learning_rate':
            self.set_lr(num(val))
        else:
            cmd = 'self.cfg.{} = {}'.format(key, val)
            try:
                exec(cmd)
                self.reset()
            except Exception as e:
                p('Failed to set configuration key "{}" to "{}"'.format(key, val))

        return self

    #----------------------------------------------------------------------------------------------------------
    # Neural Network
    #----------------------------------------------------------------------------------------------------------

    def __generate(self, x, y, extras, y_hat):
        result = None

        if has(self.event_handlers, TrainerEvents.GENERATE.value):
            result = get(self.event_handlers, TrainerEvents.GENERATE.value)(x, y, extras, y_hat)
        else:
            labels_axis = get(self.cfg, TrainerOptions.GENERATE_AXIS.value, default=1)
            result = predictions.data.max(1, keepdim=True)[1].cpu().numpy().flatten()

        return result

    def __post_test(self, results):
        if has(self.event_handlers, TrainerEvents.POST_TEST.value):
            return get(self.event_handlers, TrainerEvents.POST_TEST.value)(results)
        return results

    def _match(self, mode, x, y, extras, y_hat):
        match_results = None

        if has(self.event_handlers, TrainerEvents.MATCH_RESULTS.value):
            match_results = get(self.event_handlers, TrainerEvents.MATCH_RESULTS.value)(mode, x, y, extras, y_hat)
        else:
            match_results = self.__default_match(y_hat, y)  # Compute losses

        return match_results

    def __default_match(self, y_hat, y):
        predictions = y_hat.data.max(1, keepdim=True)[1]
        expectations = y.long()

        if torch.cuda.is_available():
            return predictions.eq(expectations.cuda())
        else:
            return predictions.cpu().eq(expectations)

    def __compute_loss(self, mode, x, y, extras, y_hat, logger):
        loss = None
        if has(self.event_handlers, TrainerEvents.COMPUTE_LOSS.value):
            loss = get(self.event_handlers, TrainerEvents.COMPUTE_LOSS.value)(mode, x, y, extras, y_hat)
        else:
            loss = self.loss_fn(y_hat, to_variable(y).long().squeeze())  # Compute losses

        extra_log_msg = {}
        if has(self.event_handlers, TrainerEvents.EXTRA_LOG_MSG.value):
            result = get(self.event_handlers, TrainerEvents.EXTRA_LOG_MSG.value)(mode, x, y, extras, y_hat)
            if result is not None:
                extra_log_msg = result

        logger.log_loss(loss.data.cpu().numpy(), **extra_log_msg)
        return loss

    def __propagate_loss(self, mode, x, y, extras, y_hat, logger):
        loss = self.__compute_loss(mode, x, y, extras, y_hat, logger)
        loss.backward()
        self.optimizer.step()

        return loss

    def __get_validation_results(self, batch_count=-1):
        dataloader = self.__get_dataloader(DEV, False)

        mode = Mode.VALIDATE
        validate_logger = Logger(self)
        validate_logger.start(mode)
        validate_logger.start_epoch()

        for i, batch in enumerate(dataloader):
            x, y, extras, y_hat = self.__process_batch(batch, validate_logger, mode)
            self.__print_batch(mode, x, y, extras, y_hat, validate_logger)

            if batch_count > 0 and i + 1 == batch_count:
                break

        percentage, (_, _, loss) = validate_logger.get_percentage(), validate_logger.get_loss()
        return percentage, loss

    def __lr_changed(self, old_lr, new_lr):
        eps = 1e-6
        for i in range(len(old_lr)):
            old, new = old_lr[i], new_lr[i]
            if old - new > eps:
                return True
        return False

    def __tune_lr(self):
        if self.scheduler is None:
            return

        precentage, loss = 0.0, 0.0
        use_train_data = get(self.cfg, TrainerOptions.SCHEDULE_ON_TRAIN_DATA.value, default=False)
        if use_train_data:
            percentage, (_, _, loss) = self.logger.get_percentage(), self.logger.get_loss()
        else:
            batch_count = get(self.cfg, TrainerOptions.SCHEDULE_BATCH_COUNT.value, default=-1)
            percentage, loss = self.__get_validation_results(batch_count)

        old_lr = self.get_lr()
        use_percentage = get(self.cfg, TrainerOptions.SCHEDULE_ON_ACCURACY.value, default=False)
        value = percentage if use_percentage else loss
        args, kwargs = filter_args(self.scheduler.step, [value], {})
        self.scheduler.step(*args, **kwargs)
        new_lr = self.get_lr()

        verbose = get(self.cfg, TrainerOptions.SCHEDULE_VERBOSE.value, default=False)
        if verbose:
            data_type = 'percentage {:.2f} %' if use_percentage else 'loss {:.8f}'
            data_source = 'training' if use_train_data else 'validation'
            template = 'Tuning learning rate using {} from {} data.'.format(data_type, data_source)
            p(template.format(value), debug=False)

        if self.__lr_changed(old_lr, new_lr):
            p('Learning rate is now at: {}'.format(new_lr))

    def __process_batch(self, batch, logger, mode=Mode.TRAIN):
        logger.increment()

        x, y, extras = batch[0], batch[1], batch[2:]
        self.optimizer.zero_grad()

        if has(self.event_handlers, TrainerEvents.PRE_PROCESS.value):
            x, y, extras = get(self.event_handlers, TrainerEvents.PRE_PROCESS.value)(mode, x, y, extras)

        if mode is Mode.TRAIN:
            self.model.train()
        else:
            self.model.eval()

        y_hat = None
        if has(self.event_handlers, TrainerEvents.MODEL_EXTRA_ARGS.value):
            args, kwargs = get(self.event_handlers, TrainerEvents.MODEL_EXTRA_ARGS.value)(mode, x, y, extras)
            y_hat = forward(self.model, [to_variable(x)] + args, kwargs)
        else:
            y_hat = self.model(to_variable(x))

        if has(self.event_handlers, TrainerEvents.POST_PROCESS.value):
            y_hat = get(self.event_handlers, TrainerEvents.POST_PROCESS.value)(mode, x, y, extras, y_hat)

        if mode is Mode.TRAIN:
            self.__propagate_loss(mode, x, y, extras, y_hat, logger)
        elif mode is Mode.VALIDATE:
            self.__compute_loss(mode, x, y, extras, y_hat, logger)

        return x, y, extras, y_hat

    def __print_batch(self, mode, x, y, extras, y_hat, logger):
        logger.log_batch(mode, x, y, extras, y_hat)
        logger.print_batch(logger is self.logger)

    def __save_path(self, save_as):
        folder = os.path.join(csd(), self.submissions_folder, self.name)
        file = None

        if save_as == SaveAs.CSV:
            file = '{} - {:03d}.csv'.format(self.current_cfg, self.epoch_ran)
        elif save_as == SaveAs.NPY:
            file = '{} - {:03d}.npy'.format(self.current_cfg, self.epoch_ran)

        return folder, file

    def __save_results(self, results, save_as):
        folder, file = self.__save_path(save_as)
        if folder is None or file is None:
            return

        path = os.path.join(folder, file)
        mkdirp(folder)

        if save_as == SaveAs.CSV:
            field_names = get(self.cfg, TrainerOptions.CSV_FIELD_NAMES.value, default=['id', 'label'])
            write_to_csv(results, path, field_names)
        elif save_as == SaveAs.NPY:
            np.save(path, np.array(results, dtype='object'))

        p('Submission file saved to "{}".'.format(path))

    def run(self, epochs=1):
        has_scheduler = self.scheduler != None
        schedule_on_batch = get(self.cfg, TrainerOptions.SCHEDULE_ON_BATCH.value, default=False)
        schedule_first = get(self.cfg, TrainerOptions.SCHEDULE_FIRST.value, default=True)

        dev_mode = get(self.cfg, TrainerOptions.DEV_MODE.value, default=False)
        train_type = DEV if dev_mode else TRAIN
        dataloader = self.__get_dataloader(train_type)

        self.logger.start()
        for epoch in range(epochs):
            self.logger.start_epoch()

            if has_scheduler and not schedule_on_batch and schedule_first:
                self.__tune_lr()

            for batch in dataloader:
                if has_scheduler and schedule_on_batch and schedule_first:
                    self.__tune_lr()

                x, y, extras, y_hat = self.__process_batch(batch, self.logger)

                if has_scheduler and schedule_on_batch and not schedule_first:
                    self.__tune_lr()

                self.__print_batch(Mode.TRAIN, x, y, extras, y_hat, self.logger)

            if has_scheduler and not schedule_on_batch and not schedule_first:
                self.__tune_lr()

            self.epoch_ran += 1
            percentage, (_, loss, _) = self.logger.get_percentage(), self.logger.get_loss()
            if abs(percentage) < 1e-6:
                percentage = None
            self.save_model(percentage, loss)

        return self

    def validate(self):
        return self.test(Mode.VALIDATE)

    def test(self, mode=Mode.TEST):
        data_type = TEST if mode == Mode.TEST else DEV
        dataloader = self.__get_dataloader(data_type)

        self.logger.start(mode)
        self.logger.start_epoch()

        results = []
        for batch in dataloader:
            x, y, extras, y_hat = self.__process_batch(batch, self.logger, mode)

            if mode == Mode.TEST:
                result = self.__generate(x, y, extras, y_hat)

                for i in range(len(result)):
                    results.append(result[i])

            self.__print_batch(mode, x, y, extras, y_hat, self.logger)

        if mode is Mode.TEST:
            results = self.__post_test(results)
            save_as = get(self.cfg, TrainerOptions.SAVE_AS.value, default=SaveAs.CSV)
            self.__save_results(results, save_as)
        else:
            self.logger.print_summary()

        return self
