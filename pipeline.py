import logging
import train
import generate
import utils


_logger = logging.getLogger(__name__)


def main(folder, batches, batch_size, workers, redis_address, starting_generation, final_generation):
    for i in range(starting_generation, final_generation):
        _logger.info(f'self-play started({batch_size}): {i}')
        generate.main(
            'gen-' + str(i),
            800,
            1.0,
            folder,
            batches,
            batch_size,
            workers,
            redis_address
        )
        _logger.info(f'self-play finished({batch_size}): {i}')
        _logger.info(f'training started: {i} -> {i + 1}')
        train.main(
            folder,
            'gen-' + str(i),
            'gen-' + str(i + 1),
            64,
            5000
        )
        _logger.info(f'training finished: {i} -> {i + 1}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folder', '-o',
        type=str,
        default='',
        help='Root folder.'
    )
    parser.add_argument(
        '--batches', '-b',
        type=int,
        default=64,
        help='Number of batches. Each batch of games will be saved separately.'
    )
    parser.add_argument(
        '--batch_size', '-n',
        type=int,
        default=112,
        help='The size of each batch. This many games will be played in a batch.'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='The number of workers.'
    )
    parser.add_argument(
        '--starting_generation', '-sg',
        type=int,
        default=0,
        help='Start self-play, training from this generation.'
    )
    parser.add_argument(
        '--final_generation', '-fg',
        type=int,
        default=50,
        help='Stop self-play, training when this generation is reached.'
    )
    parser.add_argument(
        '--redis_address', '-r',
        type=str,
        default=None,
        help='Head node\'s redis address "host:port". If not set workers will be spawned on the host machine.'
    )

    flags, _ = parser.parse_known_args()

    utils.register_logger(_logger, flags.folder, 'pipeline.log')

    main(flags.folder,
         flags.batches,
         flags.batch_size,
         flags.workers,
         flags.redis_address,
         flags.starting_generation,
         flags.final_generation)
