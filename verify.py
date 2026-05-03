import sys
sys.argv = ['x','--model','CNN_DEEP_MULTI','--trainer','TDA','--with_policy']
from trainer.multi_head import MultiHeadTrainer
import inspect

print('Class inheritance:', MultiHeadTrainer.__bases__)

required_methods = ['__init__', '_play', 'play_game', 'batch_trainer', '_train']
for method in required_methods:
    has_method = hasattr(MultiHeadTrainer, method)
    status = 'OK' if has_method else 'MISSING'
    print(f'  {method}: {status}')

sig_init = inspect.signature(MultiHeadTrainer.__init__)
print(f'__init__ signature: {sig_init}')

sig_play = inspect.signature(MultiHeadTrainer._play)
print(f'_play signature: {sig_play}')

sig_play_game = inspect.signature(MultiHeadTrainer.play_game)
print(f'play_game signature: {sig_play_game}')

sig_batch = inspect.signature(MultiHeadTrainer.batch_trainer)
print(f'batch_trainer signature: {sig_batch}')
