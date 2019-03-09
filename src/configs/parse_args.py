import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--seed', dest='seed', type=int, default=1)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--train_rollout_repeat', '-trr', type=int, default=1,
                        help='repeat rollout n times in training (mimic n * num_envs agents)')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--large_map_size', type=int, default=80,
                        help='large map size')
    parser.add_argument('--use_rgb_with_map', action='store_true',
                        help='use rgb image as input as well besides map')
    parser.add_argument('--start_indoor', action='store_true',
                        help='agent starts from an indoor position')
    parser.add_argument('--render_door', action='store_true',
                        help='whether to render door or not')
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='number of gpu for rendering')
    parser.add_argument('--ob_dilation_kernel', type=int, default=5,
                        help='kernel size for obstacle dilation')
    parser.add_argument('--max_depth', type=float, default=3,
                        help='depth limit for depth image, better greater than 3 (m)')
    parser.add_argument('--lr', dest='lr', type=float, default=0.00001)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--weight_decay', type=float, default=0.00,
                        help='weight_decay')
    parser.add_argument('--save_interval', type=int, default=30,
                        help='save model every n episodes/iterations')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='logging every n episodes/iterations')
    parser.add_argument('--max_iters', type=int, default=1000000,
                        help='maximum number of episodes/iterations')
    parser.add_argument('--num_steps', type=int, default=1000,
                        help='number of steps in an episode')
    parser.add_argument('--noptepochs', type=int, default=5,
                        help='network training epochs in each iteration')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='gamma to calculate return')
    parser.add_argument('--fix_cnn', action='store_true',
                        help='fix cnn(resnet18) weights')
    parser.add_argument('--rnn_seq_len', type=int, default=20,
                        help='sequence length in training rnn')
    parser.add_argument('--rnn_hidden_dim', type=int, default=128,
                        help='rnn hidden layer dimension')
    parser.add_argument('--rnn_type', help='lstm or gru',
                        default='gru', choices=['gru', 'lstm'], type=str)
    parser.add_argument('--rnn_num', help='number of rnn layers',
                        default=1, type=int)
    parser.add_argument('--area_reward_scale', type=float, default=0.0005,
                        help='scaling the area reward')
    parser.add_argument('--collision_penalty', type=float, default=0.006,
                        help='penalty for collision in each step')
    parser.add_argument('--step_penalty', type=float, default=0.00,
                        help='penalty for each step')
    parser.add_argument('--max_grad_norm', type=float, default=None,
                        help='maximum gradient norm')
    parser.add_argument('--test', action='store_true',
                        help='test the trained model')
    parser.add_argument('--il_pretrain', action='store_true',
                        help='load in the pretrained model')
    parser.add_argument('--pretrain_dir', type=str,
                        default='../pretrain/il_map_rgb/model')
    parser.add_argument('--save_dir', type=str, default='./data')
    parser.add_argument('--resume', '-rt', action='store_true',
                        help='resume training')
    parser.add_argument('--resume_step', '-rs', type=int, default=None)
    parser.add_argument('--render', action='store_true',
                        help='render in testing')
    parser.add_argument('--lam', type=float, default=0.95,
                        help='lambda to calculate gae')
    parser.add_argument('--cliprange', type=float, default=0.2,
                        help='clip range')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                        help='entropy loss coefficient')
    parser.add_argument('--vf_coef', type=float, default=1,
                        help='value function loss coefficient')
    return parser.parse_args()
