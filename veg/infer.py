"""Inference entry point, without opening training or validation corpora."""
import argparse


def main():
    from .veg3 import generate, load
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load', required=True, help='VEG checkpoint')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--length', type=int, default=200)
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=.8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not args.prompt or args.length < 0 or args.context < 1 or args.temperature <= 0:
        parser.error('Use a nonempty prompt, nonnegative length and positive context/temperature.')
    weights, config = load(args.load)
    print(generate(weights, config, args.prompt, args.length,
                   args.temperature, args.context, args.seed))


if __name__ == '__main__':
    main()
