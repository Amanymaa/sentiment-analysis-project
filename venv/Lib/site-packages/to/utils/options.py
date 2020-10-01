from .enum import BaseEnum, BaseIntEnum
import torch.nn as nn

#----------------------------------------------------------------------------------------------------------
# Data
#----------------------------------------------------------------------------------------------------------

data_type_name = lambda x: ['dev', 'train', 'test'][x]


class DataOptions(BaseEnum):
    Data_FOLDER = 'data_folder'  # Defaults to 'data'
    DEV_DATA_FILE = 'dev_data_file'  # Defaults to 'dev-features.npy'
    DEV_LABELS_FILE = 'dev_labels_file'  # Defaults to 'dev-labels.npy'
    TRAIN_DATA_FILE = 'train_data_file'  # Defaults to 'train-features.npy'
    TRAIN_LABELS_FILE = 'train_labels_file'  # Defaults to 'train-labels.npy'
    TEST_DATA_FILE = 'test_data_file'  # Defaults to 'test-features.npy'


#----------------------------------------------------------------------------------------------------------
# Trainer
#----------------------------------------------------------------------------------------------------------


class SaveAs(BaseIntEnum):
    CSV = 0
    NPY = 1


class TrainerOptions(BaseEnum):
    DEV_MODE = 'dev_mode'  # Defaults to False
    BATCH_SIZE = 'batch_size'  # Defaults to 64
    AUTO_RELOAD_SAVED_MODEL = 'auto_reload_saved_model'

    CFG_FOLDER = 'cfg_folder'
    MODELS_FOLDER = 'models_folder'
    SUBMISSIONS_FOLDER = 'submissions_folder'  # Defaults to 'submissions'

    OPTIMIZER = 'optimizer'  # Defaults to Adam
    OPTIMIZER_ARGS = 'optimizer_args'  # Defaults to { 'lr': 0.01 }
    LOSS_FN = 'loss_fn'  # Defaults to nn.CrossEntropyLoss

    SCHEDULER = 'scheduler'  # Defaults to schedule on validation loss
    SCHEDULER_ARGS = 'scheduler_args'
    SCHEDULER_KWARGS = 'scheduler_kwargs'
    SCHEDULE_VERBOSE = 'schedule_verbose'
    SCHEDULE_FIRST = 'schedule_first'  # Run scheduler before or after train, default: before
    SCHEDULE_BATCH_COUNT = 'schedule_batch_count'  # num of batch to use to get accuracy & loss from validation
    SCHEDULE_ON_BATCH = 'schedule_on_batch'  # Call scheduler for each batch, instead of epoch
    SCHEDULE_ON_ACCURACY = 'schedule_on_accuracy'  # Only works if print_accuracy is True
    SCHEDULE_ON_TRAIN_DATA = 'schedule_on_train_data'  # Schedule on train data instead

    PRINT_INVERVAL = 'print_inverval'  # Defaults to 100
    PRINT_ACCURACY = 'print_accuracy'  # Defaults to True

    SAVE_AS = 'save_as'  # Defaults to SaveAs.CSV
    CSV_FIELD_NAMES = 'csv_field_names'  # Defaults to ['id', 'label']

    # Generate test output (batch, 1) from y_hat (batch, classes)
    GENERATE_AXIS = 'generate_axis'  # Defaults to 1


#----------------------------------------------------------------------------------------------------------
# Initialization
#----------------------------------------------------------------------------------------------------------


class InitOptions(BaseEnum):
    """
    Defaults to:
        {
            'Conv': {
                'weight': Init.xavier_uniform(),
                'bias': Init.uniform(),
            },
            nn.Linear: {
                'weight': Init.xavier_uniform(),
                'bias': Init.uniform(),
            },
            'RNNBase': {
                'weight': Init.orthogonal(),
                'bias': Init.uniform(),
            },
        }
    """
    INIT_OPTIONS = 'init_options'


#----------------------------------------------------------------------------------------------------------
# Networks
#----------------------------------------------------------------------------------------------------------


class NeuralNetworkOptions(BaseEnum):
    IN_CHANNELS = 'in_channels'  # *MUST*
    OUT_CHANNELS = 'out_channels'  # *MUST*
    LAYERS = 'layers'


class RNNModelOptions(BaseEnum):
    IN_CHANNELS = 'in_channels'  # *MUST*
    OUT_CHANNELS = 'out_channels'  # *MUST*
    LAYERS = 'layers'
    PACK_PADDED = 'pack_padded'  # Defaults to False


class ResNetOptions(BaseEnum):
    IN_CHANNELS = 'in_channels'  # *MUST*
    OUT_CHANNELS = 'out_channels'  # *MUST*
    RESNET_TYPE = 'resnet_type'  # Load default ResNet models: (18, 34, 50, 101, 152)
    RESNET_NONLINEARITY = 'resnet_nonlinearity'  # Defaults to nn.ReLU
    RESNET_NO_FC = 'resnet_no_fc'  # Defaults to False
    RESNET_DIMENSIONS = 'resnet_dimensions'  # Defaults to 2
    """
    Customize ResNet, defaults to NetworkBlock
    BasicBlock -> (ResNet 18)
    BasicBlock -> (ResNet 34)
    NetworkBlock -> (ResNet 50)
    NetworkBlock -> (ResNet 101)
    NetworkBlock -> (ResNet 152)
    """
    RESNET_BLOCK = 'resnet_block'  # Customize ResNet, defaults to NetworkBlock (ResNet 152)
    """
    Customize ResNet, defaults to [3, 8, 36, 3] (ResNet 152)
    [2, 2, 2, 2] -> (ResNet 18)
    [3, 4, 6, 3] -> (ResNet 34)
    [3, 4, 6, 3] -> (ResNet 50)
    [3, 4, 23, 3] -> (ResNet 101)
    [3, 8, 36, 3] -> (ResNet 152)
    """
    RESNET_LAYERS = 'resnet_layers'
    """
    Customize in/first block, defaults to {
        'conv_kernel_size': 7,
        'conv_stride': 2,
        'conv_padding': 3,
        'maxpool_kernel_size': 3,
        'maxpool_stride': 2,
        'maxpool_padding': 1
    }
    """
    RESNET_INBLOCK = 'resnet_inblock'
    """
    Customize out/last block, defaults to {
        'adaptive_avgpool': False,  # Adaptive AvgPool or regular AvgPool
        'avgpool_kernel_size': 7,
        'avgpool_stride': 1
    }
    """
    RESNET_OUTBLOCK = 'resnet_outblock'


#----------------------------------------------------------------------------------------------------------
# Trainer Events
#----------------------------------------------------------------------------------------------------------


class TrainerEvents(BaseEnum):  # Events that can be binded
    # Called in training and validating
    # fn(mode, x, y, extra, y_hat) => extra_logs ({'name': val})
    EXTRA_LOG_MSG = 'extra_log_msg'
    # Called when dataloader is being loaded to add collate_fn and sampler.
    # fn(self.cfg, data_type, dataset) => dataloader
    CUSTOMIZE_DATALOADER = 'customize_dataloader'
    # Called before calling model
    # fn(mode, x, y, extras) => x, y, extras
    PRE_PROCESS = 'pre_process'
    # Called when calling model to get extra args and kwargss
    # fn(mode, x, y, extras) => *args, **kwargs
    MODEL_EXTRA_ARGS = 'model_extra_args'
    # Called after calling model
    # fn(mode, x, y, extras, y_hat) => y_hat
    POST_PROCESS = 'post_process'
    # Called when loss is being computed.
    # fn(mode, x, y, extras, y_hat) => loss
    COMPUTE_LOSS = 'compute_loss'
    # Called to get percentage accuracy
    # fn(mode, x, y, extras, y_hat) => match_results (ndarray, 1 if correct, 0 otherwise)
    MATCH_RESULTS = 'match_results'
    # Called after processing in each batch in TEST mode to generate output to be written to output.
    # fn(x, y, extras, y_hat) => result (This will be written to CSV file)
    GENERATE = 'generate'
    # Called after test is completed before saving to a file.
    # fn(results) => results
    POST_TEST = 'post_test'
