import argparse
import yaml


def get_opts():
    parser = argparse.ArgumentParser()

    # General Setttings
    parser.add_argument('--root_dir', type=str, default='Batman_masked_frames',
                        help='root directory of dataset')
    parser.add_argument('--canonical_dir', type=str, default=None,
                        help='directory of canonical dataset')

    # support multiple mask as input (each mask has different deformation fields)
    parser.add_argument('--mask_dir', nargs="+", type=str, default=None,
                        help='mask of the dataset')
    parser.add_argument('--dataset_name', type=str, default='video',
                        choices=['video'],
                        help='which dataset to train/val')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[842, 512],
                        help='resolution (img_w, img_h) of the full image')
    parser.add_argument('--canonical_wh', nargs="+", type=int, default=None,
                        help='default same as the img_wh, can be set to a larger range to include more content')
    parser.add_argument('--ref_idx', type=int, default=None,
                        help='manually select a frame as reference (for rigid movement)')

    # Deformation Setting
    parser.add_argument('--encode_w', default=False, action="store_true",
                        help='whether to apply warping')

    # Training Setttings

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--num_steps', type=int, default=10000,
                        help='number of training epochs')
    parser.add_argument('--valid_iters', type=int, default=30,
                        help='valid iters for each epoch')
    parser.add_argument('--valid_batches', type=int, default=0,
                        help='valid batches for each valid process')
    parser.add_argument('--save_model_iters', type=int, default=5000,
                        help='iterations to save the models')
    parser.add_argument('--gpus', nargs="+", type=int, default=[0],
                        help='gpu devices')

    # Test Setttings
    parser.add_argument('--test', default=False, action="store_true",
                        help='whether to disable identity')

    # Model Save and Load
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained model weight to load (do not load optimizers, etc)')
    parser.add_argument('--model_save_path', type=str, default='ckpts',
                        help='save checkpoint to')
    parser.add_argument('--log_save_path', type=str, default='logs',
                        help='save log to')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    # Optimize Settings
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly', 'exponential'])

    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int,
                        default=[2500, 5000, 7500],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.5,
                        help='learning rate decay amount')

    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')

    ##### annealed positional encoding ######
    parser.add_argument('--annealed', default=False, action="store_true",
                        help='whether to apply annealed positional encoding (Only in the warping field)')
    parser.add_argument('--annealed_begin_step', type=int, default=0,
                        help='annealed step to begin for positional encoding')
    parser.add_argument('--annealed_step', type=int, default=5000,
                        help='maximum annealed step for positional encoding')

    ##### Additional losses ######
    parser.add_argument('--bg_loss', type=float, default=None,
                        help='regularize the rest part of each object ')
    parser.add_argument('--grad_loss', type=float, default=0.1,
                        help='image gradient loss weight')
    parser.add_argument('--ref_step', type=int, default=-1,
                        help='Step to stop reference frame loss.')
    parser.add_argument('--self_bg', type=bool_parser, default=False,
                        help='Whether to use self background as bg loss.')

    ##### Special cases: for black-dominated images
    parser.add_argument('--sigmoid_offset', type=float, default=0,
                        help='whether to process balck-dominated images.')

    # Other miscellaneous settings.
    parser.add_argument('--save_deform', type=bool_parser, default=False,
                        help='Whether to save deformation field or not.')
    parser.add_argument('--save_video', type=bool_parser, default=True,
                        help='Whether to save video or not.')
    parser.add_argument('--fps', type=int, default=30,
                        help='FPS of the saved video.')

    # Network settings for PE.
    parser.add_argument('--deform_D', type=int, default=6,
                        help='The depth of deformation field MLP.')
    parser.add_argument('--deform_W', type=int, default=128,
                        help='The width of deformation field MLP.')
    parser.add_argument('--vid_D', type=int, default=8,
                        help='The depth of implicit video MLP.')
    parser.add_argument('--vid_W', type=int, default=256,
                        help='The width of implicit video MLP.')
    parser.add_argument('--N_vocab_w', type=int, default=200,
                        help='number of vocabulary for warp code in the dataset for nn.Embedding')
    parser.add_argument('--N_w', type=int, default=8,
                        help='embeddings size for warping')
    parser.add_argument('--N_xyz_w', nargs="+", type=int, default=[8, 8],
                        help='positional encoding frequency of deformation field')

    # Network settings for Hash, please see details in configs/hash.json
    parser.add_argument('--vid_hash', type=bool_parser, default=False,
                        help='Whether to use hash encoding in implicit video system.')
    parser.add_argument('--deform_hash', type=bool_parser, default=False,
                        help='Whether to use hash encoding in deformation field.')

    # Config files
    parser.add_argument('--config', type=str, default=None,
                        help='path to the YAML config file.')

    args = parser.parse_args()

    if args.config is not None:
        print(args.config)
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        args_dict = vars(args)
        args_dict.update(config)
        args_new = argparse.Namespace(**args_dict)
        return args_new

    return args


def bool_parser(arg):
    """Parses an argument to boolean."""
    if isinstance(arg, bool):
        return arg
    if arg is None:
        return False
    if arg.lower() in ['1', 'true', 't', 'yes', 'y']:
        return True
    if arg.lower() in ['0', 'false', 'f', 'no', 'n']:
        return False
    raise ValueError(f'`{arg}` cannot be converted to boolean!')
